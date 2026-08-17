"""SoftCLIP -- KL against an externally supplied similarity matrix.

Instead of a one-hot target, each anchor gets a *distribution* over the in-batch
candidates, obtained by softmaxing a row of a fixed ``(N, N)`` similarity matrix
at a target temperature.  Two sources of that matrix are interesting here, and
they are scientifically very different:

**Behavioral (preferred).**  The OSF companion release (jvkqa) carries, for each
of the 90 videos, the counts of 350 raters choosing
Neutral / Pleasant / Unpleasant / Painful.  Turning those response distributions
into a video-by-video similarity gives targets defined by *human perception*,
which is the thing the EEG is supposed to encode.  Use
:meth:`SoftCLIP.from_rater_counts`.  It also sidesteps the failure mode of the
second source.

**Video-encoder self-similarity.**  Cheap and always available, but it makes the
objective partly circular: the EEG encoder is rewarded for reproducing whatever
geometry the frozen video tower already has, including its collapse modes.  Fine
as an ablation, weak as a headline.

Rater disagreement is a first-class citizen: ``row_weights`` downweights anchors
whose video the raters could not agree on, and
:meth:`SoftCLIP.disagreement_weights` computes them from the same counts.  The
prediction that normative-label alignment is *weakest* on high-disagreement
videos is a check on the typicality confound, so keep the per-row terms
loggable rather than silently averaged away.

Note that SoftCLIP needs no false-negative masking.  Two trials of the same
condition receive identical target rows, so the soft target assigns them equal
mass automatically -- the duplicate problem that
:mod:`tactus.losses.masked_infonce` patches does not arise.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .base import (
    ContrastiveLoss,
    TemperatureMixin,
    get_meta,
    mask_value,
    masked_log_softmax,
    register_loss,
    to_float,
)

#: matrix size -> (meta key, offset from id to row index)
_SIZE_TO_KEY = {360: ("condition_id", 0), 90: ("video_id", -1)}
_KEY_TO_OFFSET = {"condition_id": 0, "video_id": -1}

MatrixSpec = Union[str, "os.PathLike[str]", Sequence, torch.Tensor]


def _load_matrix(spec: MatrixSpec) -> torch.Tensor:
    """Load a square similarity matrix from a path, array, or tensor."""
    if isinstance(spec, torch.Tensor):
        mat = spec.detach().clone().float()
    elif isinstance(spec, (str, os.PathLike)):
        path = str(spec)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            import numpy as np

            mat = torch.from_numpy(np.load(path)).float()
        elif ext == ".npz":
            import numpy as np

            with np.load(path) as bundle:
                for key in ("target", "sim", "similarity", "matrix"):
                    if key in bundle:
                        mat = torch.from_numpy(bundle[key]).float()
                        break
                else:
                    first = list(bundle.keys())[0]
                    mat = torch.from_numpy(bundle[first]).float()
        elif ext in (".pt", ".pth"):
            obj = torch.load(path, map_location="cpu")
            mat = (obj if isinstance(obj, torch.Tensor) else obj["target"]).float()
        else:
            raise ValueError(
                f"unsupported target_matrix extension {ext!r}; use .npy/.npz/.pt"
            )
    else:
        mat = torch.as_tensor(spec, dtype=torch.float32)

    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"target matrix must be square (N, N), got {tuple(mat.shape)}"
        )
    return mat


def _load_vector(spec: MatrixSpec) -> torch.Tensor:
    """Load a 1-D weight vector from a path, array, or tensor."""
    if isinstance(spec, torch.Tensor):
        return spec.detach().clone().float().reshape(-1)
    if isinstance(spec, (str, os.PathLike)):
        path = str(spec)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            import numpy as np

            return torch.from_numpy(np.load(path)).float().reshape(-1)
        if ext == ".npz":
            import numpy as np

            with np.load(path) as bundle:
                for key in ("weights", "row_weights", "w"):
                    if key in bundle:
                        return torch.from_numpy(bundle[key]).float().reshape(-1)
                first = list(bundle.keys())[0]
                return torch.from_numpy(bundle[first]).float().reshape(-1)
        if ext in (".pt", ".pth"):
            obj = torch.load(path, map_location="cpu")
            t = obj if isinstance(obj, torch.Tensor) else obj["weights"]
            return t.float().reshape(-1)
        raise ValueError(
            f"unsupported row_weights extension {ext!r}; use .npy/.npz/.pt"
        )
    return torch.as_tensor(spec, dtype=torch.float32).reshape(-1)


@register_loss("softclip")
class SoftCLIP(ContrastiveLoss, TemperatureMixin):
    """Cross-modal KL against softened external similarity targets.

    Parameters
    ----------
    target_matrix
        ``(360, 360)`` (condition-level) or ``(90, 90)`` (video-level) matrix, or
        a path to ``.npy`` / ``.npz`` / ``.pt``.  Higher values mean more
        similar.  Rows may contain NaN for missing pairs; those columns are
        dropped from the target distribution.
    key
        Meta key indexing the matrix.  Inferred from the matrix size when
        omitted (360 -> ``condition_id``, 90 -> ``video_id``).
    target_temperature
        Temperature applied to the target row before softmax.  Small values
        sharpen toward one-hot (recovering InfoNCE in the limit); large values
        flatten toward uniform.  This is the single most important knob here and
        should be tuned against a held-out fold, not guessed.
    standardize_targets
        Z-score each target row over the in-batch candidates before the softmax.
        Makes ``target_temperature`` comparable across matrices with different
        scales (rater agreement in [0, 1] vs cosine in [-1, 1]).
    hard_mix
        Blend weight toward the one-hot target: ``q = (1 - a) * soft + a * hard``.
        0 is pure SoftCLIP; 1 recovers InfoNCE.
    set_diagonal
        Force the matrix diagonal to this value after loading.  Guarantees a
        sample is its own most-similar item, which some behaviorally derived
        matrices violate through estimation noise.
    row_weights
        Optional ``(N,)`` per-item weights (path or array), e.g. ``1 - normalized
        rater entropy``.  Anchors are weighted by the weight of their own item.
    symmetric
        Also apply the video-to-EEG direction.
    temperature, learnable_temperature, max_scale, min_scale
        Temperature of the *predicted* distribution.
    """

    requires_video = True

    def __init__(
        self,
        target_matrix: MatrixSpec,
        key: Optional[str] = None,
        target_temperature: float = 0.1,
        standardize_targets: bool = True,
        hard_mix: float = 0.0,
        set_diagonal: Optional[float] = None,
        row_weights: Optional[MatrixSpec] = None,
        symmetric: bool = True,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        mat = _load_matrix(target_matrix)
        n = mat.shape[0]

        if key is None:
            if n not in _SIZE_TO_KEY:
                raise ValueError(
                    f"cannot infer the meta key from a {n}x{n} matrix; pass "
                    f"key='condition_id' or key='video_id' explicitly. "
                    f"(Known sizes: {sorted(_SIZE_TO_KEY)}.)"
                )
            key = _SIZE_TO_KEY[n][0]
        if key not in _KEY_TO_OFFSET:
            raise ValueError(
                f"key must be one of {sorted(_KEY_TO_OFFSET)}, got {key!r}"
            )

        if set_diagonal is not None:
            mat.fill_diagonal_(float(set_diagonal))
        if target_temperature <= 0:
            raise ValueError(
                f"target_temperature must be > 0, got {target_temperature}"
            )
        if not (0.0 <= hard_mix <= 1.0):
            raise ValueError(f"hard_mix must be in [0, 1], got {hard_mix}")

        self._init_temperature(
            temperature,
            learnable=learnable_temperature,
            max_scale=max_scale,
            min_scale=min_scale,
        )
        self.register_buffer("target", mat)
        self.n_items = n
        self.key = key
        self.offset = _KEY_TO_OFFSET[key]
        self.target_temperature = float(target_temperature)
        self.standardize_targets = bool(standardize_targets)
        self.hard_mix = float(hard_mix)
        self.symmetric = bool(symmetric)
        self.renormalize = bool(renormalize)
        self.requires_meta = (key,)

        if row_weights is None:
            self.register_buffer("row_weight", torch.ones(n))
            self.has_row_weights = False
        else:
            w = _load_vector(row_weights)
            if w.numel() != n:
                raise ValueError(
                    f"row_weights has {w.numel()} entries, expected {n} "
                    f"(one per {self.key} item)."
                )
            self.register_buffer("row_weight", w.clamp_min(0.0))
            self.has_row_weights = True

    # -- target construction helpers --------------------------------------- #

    @staticmethod
    def from_rater_counts(
        counts, metric: str = "bhattacharyya", eps: float = 1e-8
    ) -> torch.Tensor:
        """Build an ``(N, N)`` similarity matrix from rater response counts.

        Parameters
        ----------
        counts
            ``(N, C)`` non-negative counts; for ds005662 this is the 90x4 table
            of Neutral / Pleasant / Unpleasant / Painful tallies over 350 raters.
        metric
            ``"bhattacharyya"`` (default): ``sum_c sqrt(p_c q_c)``, in [0, 1],
            the natural affinity between two categorical distributions.
            ``"js"``: ``1 - JensenShannon(p, q)`` with log base 2, also in [0, 1].
            ``"cosine"``: cosine between the normalized count vectors.

        Returns
        -------
        A symmetric matrix with values in [0, 1] and a diagonal of 1.
        """
        p = torch.as_tensor(counts, dtype=torch.float32)
        if p.dim() != 2:
            raise ValueError(f"counts must be (N, C), got {tuple(p.shape)}")
        p = p.clamp_min(0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)

        if metric == "bhattacharyya":
            r = p.sqrt()
            sim = r @ r.transpose(0, 1)
        elif metric == "cosine":
            r = p / p.norm(dim=1, keepdim=True).clamp_min(eps)
            sim = r @ r.transpose(0, 1)
        elif metric == "js":
            pi = p.unsqueeze(1)  # (N, 1, C)
            pj = p.unsqueeze(0)  # (1, N, C)
            m = 0.5 * (pi + pj)

            def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # xlogy keeps 0 * log 0 == 0 instead of NaN
                return (
                    torch.xlogy(a, a.clamp_min(eps)) - torch.xlogy(a, b.clamp_min(eps))
                ).sum(-1)

            jsd = 0.5 * _kl(pi, m) + 0.5 * _kl(pj, m)  # (N, N), nats
            sim = 1.0 - (jsd / math.log(2.0)).clamp(0.0, 1.0)
        else:
            raise ValueError(
                f"metric must be 'bhattacharyya', 'js' or 'cosine', got {metric!r}"
            )
        sim = 0.5 * (sim + sim.transpose(0, 1))  # kill float asymmetry
        sim.fill_diagonal_(1.0)
        return sim.clamp(0.0, 1.0)

    @staticmethod
    def disagreement_weights(counts, eps: float = 1e-8) -> torch.Tensor:
        """``1 - normalized entropy`` of each row of the rater count table.

        High-agreement videos get a weight near 1, videos the 350 raters split
        on get a weight near 0.  Feed as ``row_weights`` to downweight anchors
        whose normative label is poorly defined.
        """
        p = torch.as_tensor(counts, dtype=torch.float32).clamp_min(0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
        h = -torch.xlogy(p, p.clamp_min(eps)).sum(dim=1)
        h_max = float(torch.log(torch.tensor(float(p.shape[1]))))
        return (1.0 - h / max(h_max, eps)).clamp(0.0, 1.0)

    # -- forward ------------------------------------------------------------ #

    def _target_rows(self, idx: torch.Tensor) -> torch.Tensor:
        """``(B, B)`` submatrix ``T[idx_i, idx_j]``."""
        return self.target.index_select(0, idx).index_select(1, idx)

    def _soft_targets(self, tsim: torch.Tensor) -> tuple:
        """Row-softmax the target similarities; returns ``(q, valid)``."""
        valid = torch.isfinite(tsim)
        t = torch.nan_to_num(tsim, nan=0.0, posinf=0.0, neginf=0.0)
        if self.standardize_targets:
            cnt = valid.sum(dim=1, keepdim=True).clamp_min(1)
            mean = (t * valid).sum(dim=1, keepdim=True) / cnt
            var = (((t - mean) * valid) ** 2).sum(dim=1, keepdim=True) / cnt
            t = (t - mean) / var.sqrt().clamp_min(1e-6)
        logits = t / self.target_temperature
        q = torch.softmax(logits.masked_fill(~valid, mask_value(logits.dtype)), dim=1)
        return q, valid

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

        ids = get_meta(meta, self.key, device=device, batch_size=b)
        assert ids is not None
        idx = ids + self.offset
        if bool(((idx < 0) | (idx >= self.n_items)).any()):
            raise ValueError(
                f"meta['{self.key}'] maps to target-matrix rows outside "
                f"[0, {self.n_items - 1}]."
            )
        if self.target.device != device:
            self.target = self.target.to(device)
            self.row_weight = self.row_weight.to(device)

        scale = self.scale()
        logits = scale * (z_eeg @ z_vid.transpose(0, 1))  # (B, B)
        tsim = self._target_rows(idx)  # (B, B)

        q, valid = self._soft_targets(tsim)
        if self.hard_mix > 0.0:
            hard = torch.eye(b, device=device, dtype=q.dtype)
            q = (1.0 - self.hard_mix) * q + self.hard_mix * hard

        w = self.row_weight.index_select(0, idx)  # (B,)
        w_sum = w.sum().clamp_min(1e-8)

        def _kl_dir(lg: torch.Tensor, tgt: torch.Tensor, msk: torch.Tensor):
            logp = masked_log_softmax(lg, msk, dim=1)
            per_row = (torch.xlogy(tgt, tgt) - tgt * logp).sum(dim=1)  # (B,)
            return (per_row * w).sum() / w_sum, per_row

        loss_e2v, per_e2v = _kl_dir(logits, q, valid)
        if self.symmetric:
            q_t, valid_t = self._soft_targets(tsim.transpose(0, 1))
            if self.hard_mix > 0.0:
                hard = torch.eye(b, device=device, dtype=q_t.dtype)
                q_t = (1.0 - self.hard_mix) * q_t + self.hard_mix * hard
            loss_v2e, _ = _kl_dir(logits.transpose(0, 1), q_t, valid_t)
            loss = 0.5 * (loss_e2v + loss_v2e)
        else:
            loss_v2e = torch.zeros((), device=device)
            loss = loss_e2v

        with torch.no_grad():
            target_entropy = -(torch.xlogy(q, q)).sum(dim=1).mean()
            diag_mass = q.diagonal().mean()
            acc = (logits.argmax(dim=1) == torch.arange(b, device=device)).float().mean()

        return {
            "loss": loss,
            "logs": {
                "loss": to_float(loss),
                "loss_e2v": to_float(loss_e2v),
                "loss_v2e": to_float(loss_v2e),
                "logit_scale": to_float(scale),
                "temperature": self.temperature,
                "target_temperature": self.target_temperature,
                "target_entropy": to_float(target_entropy),
                "target_diag_mass": to_float(diag_mass),
                "acc_e2v": to_float(acc),
                "mean_row_weight": to_float(w.mean()),
                "n_valid": float(b),
                "degenerate": 0.0,
            },
        }

    def extra_repr(self) -> str:
        return (
            f"key={self.key}, n_items={self.n_items}, "
            f"target_temperature={self.target_temperature}, "
            f"hard_mix={self.hard_mix}, symmetric={self.symmetric}, "
            f"row_weights={self.has_row_weights}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)

    # 1) behaviorally derived targets from fake 350-rater counts (90 videos x 4)
    counts = torch.randint(0, 200, (90, 4)).float()
    beh = SoftCLIP.from_rater_counts(counts, metric="bhattacharyya")
    wts = SoftCLIP.disagreement_weights(counts)
    print(
        f"behavioral target: shape={tuple(beh.shape)} "
        f"range=[{beh.min():.3f}, {beh.max():.3f}] diag={beh.diagonal().mean():.3f}"
    )
    print(f"disagreement weights: mean={wts.mean():.3f} min={wts.min():.3f}")

    for tag, fn in [
        ("video-level/behavioral", SoftCLIP(beh, row_weights=wts)),
        ("video-level/js", SoftCLIP(SoftCLIP.from_rater_counts(counts, metric="js"))),
        (
            "condition-level/encoder",
            SoftCLIP(torch.rand(360, 360).clamp(0, 1), set_diagonal=1.0),
        ),
        ("hard-mix-0.5", SoftCLIP(beh, hard_mix=0.5)),
    ]:
        for btag, kw in [("random", {}), ("many-dups", {"n_unique_conditions": 4})]:
            ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
            out = fn(ze, zv, m)
            out["loss"].backward()
            print(
                f"[softclip/{tag:24s}/{btag:10s}] loss={out['loss'].item():.4f} "
                f"H(q)={out['logs']['target_entropy']:.3f} "
                f"diag_mass={out['logs']['target_diag_mass']:.3f}"
            )
