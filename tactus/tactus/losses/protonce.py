"""Prototype contrast -- the PRIMARY TACTUS objective.

Single-trial EEG is contrasted against an EMA bank of prototypes held at two
granularities simultaneously:

``condition`` 360 slots, ``slot = condition_id``
    Orientation-specific.  Keeps the mirror-discriminative signal that the
    equivariance analysis needs.
``video``      90 slots, ``slot = video_id - 1``
    Pools the four orientations of a clip, so its prototype is the
    orientation-averaged embedding.  Contrasting against it *is* the
    orientation-invariance objective.

Running both at once, with separate weights, is how the orientation policy fork
is expressed as a continuum rather than a binary switch.

Why a bank rather than in-batch negatives
-----------------------------------------
The retrieval metric asks a single trial to pick its condition out of a gallery
of up to 360.  An in-batch objective can only ever present ``B - 1`` negatives
and, worse, *which* negatives it presents depends on the sampler, so the loss
scale drifts with batch composition.  A prototype bank presents the whole
gallery every step, at fixed cost, and is completely insensitive to how many
duplicate conditions the sampler happened to draw -- which is the exact failure
mode that forces the masking machinery in ``masked_infonce``.  Single-trial SNR
is the binding constraint in this dataset; not wasting negatives on batch
accidents matters.

Gradient flow
-------------
The bank is a buffer built under ``no_grad``, so the negatives are constants
(MoCo-style stop-gradient).  With ``live_positive=True`` (default) the positive
column is replaced by the *live* differentiable ``scale * <z_eeg, z_vid>``, so
gradient still reaches the video projector through the positive term.  There is
no repulsive gradient on the projector, which is by design -- pair this loss
with a small in-batch term via :class:`tactus.losses.composite.CompositeLoss`
if the projector needs one.

Evaluation on unseen videos
---------------------------
Held-out base videos have no prototype: their slots are uninitialized and are
masked out of the denominator.  Rows whose *own* slot is uninitialized are
dropped from the mean (and counted in ``frac_unseen``) unless ``live_positive``
supplies the positive directly.  For zero-shot retrieval, inject the held-out
gallery explicitly before evaluating::

    loss_fn.set_prototypes(cond_ids, projector(video_emb), granularity="condition")

The bank is never updated while ``self.training`` is False (unless
``update_in_eval=True``), so evaluation cannot contaminate it.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from .base import (
    ContrastiveLoss,
    TemperatureMixin,
    get_meta,
    masked_log_softmax,
    register_loss,
    safe_normalize,
    to_float,
)

#: granularity -> (meta key, additive offset from id to slot index)
_GRAN_SPEC: Dict[str, tuple] = {
    "condition": ("condition_id", 0),
    "video": ("video_id", -1),
}

_VALID_SOURCES = ("video", "eeg", "both")


@register_loss("protonce")
class ProtoNCE(ContrastiveLoss, TemperatureMixin):
    """EMA prototype contrast at condition and/or video granularity.

    Parameters
    ----------
    dim
        Embedding dimension.  May be ``None``, in which case the banks are
        allocated on the first forward pass; set it explicitly in configs so
        that checkpoint loading never depends on call order.
    n_conditions, n_videos, n_orientations
        Sizes of the two banks and the orientation factor (used for sibling
        masking).  Defaults match ds005662 exactly.
    granularities
        Subset of ``("condition", "video")``.
    weights
        Per-granularity loss weights.  Defaults to
        ``{"condition": 1.0, "video": 0.5}`` when both granularities are active
        (condition-level is the primary objective), and to 1.0 when only one is.
    momentum
        EMA coefficient ``m`` in ``p <- m * p + (1 - m) * batch_mean``.  0.99 is
        a good default at ~2 k steps/epoch; lower it if a slot is visited rarely
        (a 360-slot bank with batch 256 sees each slot roughly once per 1.4
        steps at uniform sampling, but stratified samplers skew this).
    source
        Which embeddings feed the bank: ``"video"`` (default -- prototypes track
        the projected frozen codebook), ``"eeg"`` (class centroids of the EEG
        embedding, a CLISA-like objective), or ``"both"`` (normalized mean).
    live_positive
        Replace the positive logit with the differentiable in-batch
        ``<z_eeg, z_vid>``.  Required if the video projector is to receive
        gradient from this loss; automatically disabled when ``source="eeg"``.
    cold_start_fill
        Before computing the loss, fill *uninitialized* slots from the current
        batch so the first steps are not wasted.  Already-warm slots are
        untouched, so this leaks nothing that the EMA would not have written a
        step later.
    min_count
        A slot must have absorbed at least this many samples before it is
        allowed into the denominator.  Raising it suppresses noisy half-formed
        prototypes early in training.
    mask_sibling_orientations
        At ``condition`` granularity, drop the anchor's other three orientation
        slots from the denominator.  The "flips are not negatives" position.
    update_in_eval
        Keep updating the bank when ``self.training`` is False.  Almost always
        wrong; exposed for online-adaptation experiments only.
    label_smoothing
        Optional smoothing over the available slots.
    """

    requires_video = True

    def __init__(
        self,
        dim: Optional[int] = None,
        n_conditions: int = 360,
        n_videos: int = 90,
        n_orientations: int = 4,
        granularities: Sequence[str] = ("condition", "video"),
        weights: Optional[Mapping[str, float]] = None,
        momentum: float = 0.99,
        source: str = "video",
        live_positive: bool = True,
        cold_start_fill: bool = True,
        min_count: int = 1,
        mask_sibling_orientations: bool = False,
        update_in_eval: bool = False,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        label_smoothing: float = 0.0,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        grans = tuple(str(g).lower() for g in granularities)
        bad = [g for g in grans if g not in _GRAN_SPEC]
        if bad:
            raise ValueError(
                f"unknown granularity {bad}; valid: {sorted(_GRAN_SPEC)}"
            )
        if not grans:
            raise ValueError("granularities must be non-empty")
        if source not in _VALID_SOURCES:
            raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        self._init_temperature(
            temperature,
            learnable=learnable_temperature,
            max_scale=max_scale,
            min_scale=min_scale,
        )
        self.granularities = grans
        self.n_slots = {"condition": int(n_conditions), "video": int(n_videos)}
        self.n_orientations = int(n_orientations)
        self.momentum = float(momentum)
        self.source = source
        self.live_positive = bool(live_positive) and source != "eeg"
        if self.live_positive and source == "video":
            warnings.warn(
                "ProtoNCE(live_positive=True, source='video'): the positive logit "
                "is the live differentiable <z_eeg, z_vid> while every negative is "
                "a stale detached EMA bank entry. A trainable video projector can "
                "drive this loss to zero by outrunning its own lagging bank, with "
                "no EEG-video relationship at all (verified: 100% training accuracy "
                "on pure-noise EEG, see tests/test_protonce_shortcut.py). Set "
                "live_positive=False unless you pair this with an in-batch "
                "repulsive term via CompositeLoss.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.cold_start_fill = bool(cold_start_fill)
        self.min_count = int(min_count)
        self.mask_sibling_orientations = bool(mask_sibling_orientations)
        self.update_in_eval = bool(update_in_eval)
        self.label_smoothing = float(label_smoothing)
        self.renormalize = bool(renormalize)
        # Condition-level is the primary objective; the video-level term is a
        # secondary orientation-invariance pull, so it defaults to half weight
        # rather than silently matching the main term. A single-granularity
        # instance always gets weight 1.0.
        w = dict(weights or {})
        default_w = {"condition": 1.0, "video": 0.5} if len(grans) > 1 else {}
        self.weights = {g: float(w.get(g, default_w.get(g, 1.0))) for g in grans}
        self.dim = int(dim) if dim is not None else None
        self.requires_meta = tuple(_GRAN_SPEC[g][0] for g in grans)

        d = self.dim if self.dim is not None else 0
        for g in grans:
            s = self.n_slots[g]
            self.register_buffer(f"bank_{g}", torch.zeros(s, d, dtype=torch.float32))
            self.register_buffer(f"init_{g}", torch.zeros(s, dtype=torch.bool))
            self.register_buffer(f"count_{g}", torch.zeros(s, dtype=torch.long))
        # slot -> base-video index, for sibling-orientation masking
        if "condition" in grans:
            self.register_buffer(
                "slot_video_condition",
                torch.arange(self.n_slots["condition"]) // self.n_orientations,
                persistent=False,
            )

    # -- buffer accessors --------------------------------------------------- #

    def _bank(self, g: str) -> torch.Tensor:
        return getattr(self, f"bank_{g}")

    def _init(self, g: str) -> torch.Tensor:
        return getattr(self, f"init_{g}")

    def _count(self, g: str) -> torch.Tensor:
        return getattr(self, f"count_{g}")

    def _ensure_dim(self, d: int, device: torch.device) -> None:
        """Allocate the banks on first use when ``dim`` was not given."""
        width = self._bank(self.granularities[0]).shape[1]
        if width != 0:
            if width != d:
                raise ValueError(
                    f"ProtoNCE was built with dim={width} but received embeddings "
                    f"of dim={d}. Set `dim` in the loss config to match the "
                    f"projection head."
                )
            return
        self.dim = int(d)
        for g in self.granularities:
            s = self.n_slots[g]
            setattr(self, f"bank_{g}", torch.zeros(s, d, dtype=torch.float32, device=device))
            setattr(self, f"init_{g}", torch.zeros(s, dtype=torch.bool, device=device))
            setattr(self, f"count_{g}", torch.zeros(s, dtype=torch.long, device=device))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[override]
        """Resize lazily-allocated banks to match an incoming checkpoint."""
        for g in self.granularities:
            key = f"{prefix}bank_{g}"
            if key in state_dict:
                incoming = state_dict[key]
                if self._bank(g).shape != incoming.shape:
                    setattr(self, f"bank_{g}", torch.zeros_like(incoming))
                    self.dim = int(incoming.shape[1])
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # -- slots -------------------------------------------------------------- #

    def _slots(
        self, g: str, meta: Mapping[str, torch.Tensor], b: int, device: torch.device
    ) -> torch.Tensor:
        key, offset = _GRAN_SPEC[g]
        ids = get_meta(meta, key, device=device, batch_size=b)
        assert ids is not None
        slots = ids + offset
        s = self.n_slots[g]
        if bool(((slots < 0) | (slots >= s)).any()):
            lo, hi = int(slots.min()), int(slots.max())
            raise ValueError(
                f"granularity '{g}': meta['{key}'] maps to slot range [{lo}, {hi}] "
                f"which is outside [0, {s - 1}]. Check n_conditions/n_videos, and "
                f"that {key} follows the trial-table convention "
                f"(condition_id is 0-based, video_id is 1-based)."
            )
        return slots

    def _source_emb(self, z_eeg: torch.Tensor, z_vid: torch.Tensor) -> torch.Tensor:
        if self.source == "video":
            return z_vid
        if self.source == "eeg":
            return z_eeg
        return safe_normalize(0.5 * (z_eeg + z_vid))

    # -- bank maintenance --------------------------------------------------- #

    @torch.no_grad()
    def _accumulate(
        self,
        g: str,
        slots: torch.Tensor,
        emb: torch.Tensor,
        only_uninitialized: bool = False,
    ) -> None:
        bank, init, count = self._bank(g), self._init(g), self._count(g)
        emb = emb.detach().to(bank.dtype)
        if only_uninitialized:
            keep = ~init[slots]
            if not bool(keep.any()):
                return
            slots, emb = slots[keep], emb[keep]
        if slots.numel() == 0:
            return

        uniq, inv = torch.unique(slots, return_inverse=True)
        sums = torch.zeros(
            uniq.numel(), bank.shape[1], dtype=bank.dtype, device=bank.device
        )
        sums.index_add_(0, inv, emb)
        ones = torch.ones(slots.numel(), dtype=bank.dtype, device=bank.device)
        cnt = torch.zeros(uniq.numel(), dtype=bank.dtype, device=bank.device)
        cnt.index_add_(0, inv, ones)
        means = safe_normalize(sums / cnt.unsqueeze(1))

        was_init = init[uniq].unsqueeze(1)
        blended = torch.where(
            was_init, self.momentum * bank[uniq] + (1.0 - self.momentum) * means, means
        )
        bank[uniq] = safe_normalize(blended)
        init[uniq] = True
        count[uniq] = count[uniq] + cnt.long()

    @torch.no_grad()
    def set_prototypes(
        self,
        ids: torch.Tensor,
        emb: torch.Tensor,
        granularity: str = "condition",
        mark_initialized: bool = True,
    ) -> None:
        """Overwrite prototypes for the given raw ids (not slot indices).

        Use this to install a zero-shot gallery: project the frozen video
        embeddings of held-out conditions and write them in before evaluating.
        """
        if granularity not in self.granularities:
            raise ValueError(
                f"granularity '{granularity}' not maintained by this loss "
                f"(has {self.granularities})"
            )
        bank, init, count = (
            self._bank(granularity),
            self._init(granularity),
            self._count(granularity),
        )
        if bank.shape[1] == 0:
            self._ensure_dim(emb.shape[1], emb.device)
            bank, init, count = (
                self._bank(granularity),
                self._init(granularity),
                self._count(granularity),
            )
        offset = _GRAN_SPEC[granularity][1]
        slots = torch.as_tensor(ids, device=bank.device).reshape(-1).long() + offset
        emb = safe_normalize(emb.detach().to(bank.dtype).to(bank.device))
        if slots.numel() != emb.shape[0]:
            raise ValueError(
                f"ids has {slots.numel()} entries but emb has {emb.shape[0]} rows"
            )
        bank[slots] = emb
        if mark_initialized:
            init[slots] = True
            count[slots] = count[slots].clamp_min(self.min_count)

    @torch.no_grad()
    def prototypes(self, granularity: str = "condition"):
        """Return ``(bank_copy, initialized_mask_copy)`` for external retrieval."""
        return self._bank(granularity).clone(), self._init(granularity).clone()

    @torch.no_grad()
    def reset_bank(self, granularity: Optional[str] = None) -> None:
        """Zero the bank(s).  Call between folds -- prototypes must never cross
        a video-disjoint split boundary."""
        for g in self.granularities if granularity is None else [granularity]:
            self._bank(g).zero_()
            self._init(g).zero_()
            self._count(g).zero_()

    @torch.no_grad()
    def bank_coverage(self) -> Dict[str, float]:
        """Fraction of initialized slots per granularity."""
        return {
            g: float(self._init(g).float().mean().item()) for g in self.granularities
        }

    @torch.no_grad()
    def sync_banks_(self) -> None:
        """Average banks across DDP ranks (call at epoch boundaries).

        Each rank updates its bank from its own local batches, so without this
        the ranks slowly diverge.  The divergence is usually benign (the banks
        are only providing negatives) but it makes ``bank_coverage`` and any
        bank-based retrieval log rank-dependent, which is confusing.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world = torch.distributed.get_world_size()
        for g in self.granularities:
            bank, init, count = self._bank(g), self._init(g), self._count(g)
            torch.distributed.all_reduce(bank, op=torch.distributed.ReduceOp.SUM)
            bank.div_(world)
            bank.copy_(safe_normalize(bank))
            init_f = init.float()
            torch.distributed.all_reduce(init_f, op=torch.distributed.ReduceOp.MAX)
            init.copy_(init_f > 0)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

    # -- forward ------------------------------------------------------------ #

    def _granularity_loss(
        self,
        g: str,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        slots: torch.Tensor,
        scale: torch.Tensor,
    ):
        bank, init, count = self._bank(g), self._init(g), self._count(g)
        b, device = z_eeg.shape[0], z_eeg.device

        avail = init & (count >= self.min_count)  # (S,)
        # Snapshot the bank. The matmul saves its right operand for backward, and
        # the EMA update at the end of forward writes into the bank in place --
        # without the copy, autograd raises "variable needed for gradient
        # computation has been modified by an inplace operation". The bank is a
        # constant w.r.t. autograd anyway, and 360 x D floats is a trivial copy.
        bank_snapshot = bank.detach().clone().to(z_eeg.dtype)
        logits = scale * (z_eeg @ bank_snapshot.transpose(0, 1))  # (B, S)

        cand = avail.unsqueeze(0).expand(b, -1).clone()  # (B, S)
        if g == "condition" and self.mask_sibling_orientations:
            slot_video = self.slot_video_condition  # (S,)
            anchor_video = slots // self.n_orientations  # (B,)
            cand = cand & ~(slot_video.unsqueeze(0) == anchor_video.unsqueeze(1))

        tgt = slots.unsqueeze(1)  # (B, 1)
        target_onehot = torch.zeros_like(cand).scatter_(
            1, tgt, torch.ones_like(tgt, dtype=torch.bool)
        )
        cand = cand | target_onehot  # the positive column is always in

        if self.live_positive:
            pos_logit = scale * (z_eeg * z_vid).sum(dim=-1, keepdim=True)  # (B, 1)
            logits = logits.scatter(1, tgt, pos_logit)
            pos_ok = torch.ones(b, dtype=torch.bool, device=device)
        else:
            pos_ok = avail[slots]

        has_neg = (cand & ~target_onehot).any(dim=1)
        row_valid = has_neg & pos_ok
        n_valid = row_valid.sum()

        logp = masked_log_softmax(logits, cand, dim=1)
        row = -logp.gather(1, tgt).squeeze(1)  # (B,)
        if self.label_smoothing > 0.0:
            n_cand = cand.sum(dim=1).clamp_min(1).to(logp.dtype)
            uniform = -(logp * cand.to(logp.dtype)).sum(dim=1) / n_cand
            row = (1.0 - self.label_smoothing) * row + self.label_smoothing * uniform

        if int(n_valid) == 0:
            zero = logits.sum() * 0.0 + 0.0  # graph-connected, and not -0.0
            stats = {
                "loss": 0.0,
                "acc": 0.0,
                "n_valid": 0.0,
                "coverage": to_float(avail.float().mean()),
                "frac_unseen": 1.0,
                "mean_n_cand": 0.0,
            }
            return zero, stats

        denom = n_valid.to(row.dtype)
        loss = (row * row_valid.to(row.dtype)).sum() / denom

        with torch.no_grad():
            pred = logits.masked_fill(~cand, float("-inf")).argmax(dim=1)
            acc = ((pred == slots) & row_valid).float().sum() / denom
            stats = {
                "loss": to_float(loss),
                "acc": to_float(acc),
                "n_valid": float(int(n_valid)),
                "coverage": to_float(avail.float().mean()),
                "frac_unseen": to_float((~avail[slots]).float().mean()),
                "mean_n_cand": to_float(cand.sum(dim=1).float().mean()),
            }
        return loss, stats

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        z_eeg, z_vid = self._prepare(z_eeg, z_vid, self.renormalize)
        assert z_vid is not None
        b, device = z_eeg.shape[0], z_eeg.device
        self._ensure_dim(z_eeg.shape[1], device)
        if self._bank(self.granularities[0]).device != device:
            for g in self.granularities:
                setattr(self, f"bank_{g}", self._bank(g).to(device))
                setattr(self, f"init_{g}", self._init(g).to(device))
                setattr(self, f"count_{g}", self._count(g).to(device))

        scale = self.scale()
        do_update = self.training or self.update_in_eval
        src = self._source_emb(z_eeg, z_vid)

        slots_by_g = {
            g: self._slots(g, meta, b, device) for g in self.granularities
        }

        # Cold start BEFORE the loss so step 0 is not wasted; only touches slots
        # that have never been written, so no already-warm prototype is leaked.
        if do_update and self.cold_start_fill:
            for g in self.granularities:
                self._accumulate(g, slots_by_g[g], src, only_uninitialized=True)

        total: Optional[torch.Tensor] = None
        logs: Dict[str, float] = {}
        for g in self.granularities:
            w = self.weights[g]
            loss_g, stats_g = self._granularity_loss(
                g, z_eeg, z_vid, slots_by_g[g], scale
            )
            for k, v in stats_g.items():
                logs[f"{g}_{k}"] = v
            logs[f"{g}_weight"] = w
            contrib = w * loss_g
            total = contrib if total is None else total + contrib

        # EMA update AFTER the loss: the negatives a step sees are the state of
        # the bank *before* this batch touched it.
        if do_update:
            for g in self.granularities:
                self._accumulate(g, slots_by_g[g], src, only_uninitialized=False)

        if total is None:
            total = self._zero_loss(z_eeg, z_vid)

        logs["loss"] = to_float(total)
        logs["logit_scale"] = to_float(scale)
        logs["temperature"] = self.temperature
        return {"loss": total, "logs": logs}

    def extra_repr(self) -> str:
        return (
            f"granularities={self.granularities}, weights={self.weights}, "
            f"source={self.source}, momentum={self.momentum}, "
            f"live_positive={self.live_positive}, dim={self.dim}, "
            f"temperature={self.temperature:.4f}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    fn = ProtoNCE(dim=256)
    print(fn)
    for step in range(4):
        ze, zv, m = make_dummy_batch(batch_size=32, dim=256, seed=step)
        out = fn(ze, zv, m)
        out["loss"].backward()
        print(
            f"[protonce/step{step}] loss={out['loss'].item():.4f} "
            f"cond_acc={out['logs']['condition_acc']:.3f} "
            f"cov={fn.bank_coverage()}"
        )

    # eval on an unseen condition: bank must not update, row must not NaN
    fn.eval()
    ze, zv, m = make_dummy_batch(batch_size=32, dim=256, seed=99)
    cov_before = fn.bank_coverage()
    out = fn(ze, zv, m)
    assert fn.bank_coverage() == cov_before, "bank updated during eval"
    print(f"[protonce/eval] loss={out['loss'].item():.4f} logs={out['logs']}")

    # zero-shot gallery injection
    fn.set_prototypes(
        torch.arange(360), safe_normalize(torch.randn(360, 256)), "condition"
    )
    print("[protonce/injected] coverage:", fn.bank_coverage())

    # degenerate: single condition, single sample
    for tag, kw in [("all-same-condition", {"single_condition": True}), ("B=1", {"batch_size": 1})]:
        f2 = ProtoNCE(dim=256)
        ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
        out = f2(ze, zv, m)
        out["loss"].backward()
        print(f"[protonce/{tag}] loss={out['loss'].item():.4f} finite={torch.isfinite(out['loss']).item()}")
