"""Factorized Hierarchical Multi-positive Contrastive learning (FHMC).

The merged-blueprint (v3) flagship objective.  It operationalizes the
*competing invariances* problem specific to ds005662: the four orientation
flips of a base video are simultaneously (a) augmentations of the same touch
content and (b) carriers of real perspective/laterality information.  A single
shared embedding cannot both collapse and preserve them, so the objective
factorizes the trunk space:

=========  ==============================================================
space      objective (all cross-modal EEG <-> video unless noted)
=========  ==============================================================
trunk      ``exact``     multi-positive InfoNCE, positives = same
                         ``condition_id`` (base video x orientation).  Kept in
                         the TRUNK space so the trainer's existing retrieval
                         eval and the pre-registered primary endpoint
                         (test/video/g18/top1_pseudo) measure it unchanged.
content    ``content``   multi-positive InfoNCE, positives = same ``video_id``
                         regardless of orientation (flip-invariant touch
                         content).
geometry   ``geometry``  multi-positive InfoNCE, positives = same
                         ``orientation`` across *different* videos, plus an
                         auxiliary 4-way orientation classifier (CE) on the
                         EEG side (flip-equivariant viewpoint code).
semantic   ``semantic``  distributional alignment with soft targets from the
                         continuous affect kernel
                         ``w_ij = exp(-(|dv|+|da|+|dt|) / sigma)``
                         (valence/arousal/threat, z-scored upstream).
--         ``disent``    squared cross-covariance between the batch-centred
                         content and geometry embeddings (EEG side), pushing
                         the two heads toward carrying different information.
=========  ==============================================================

Implementation notes
--------------------
* The factor heads are *inside this loss module*.  This is deliberate:
  ``tactus/train/trainer.py`` already optimizes ``loss_fn.parameters()``
  (trainer.py:932), so heads-in-loss needs zero trainer changes and the
  ``build_loss`` one-key swap contract survives.  The heads are shared across
  modalities (one linear map per factor applied to both z_eeg and z_vid),
  which is what makes each head a *subspace* of the common trunk space.
* Eval wiring: retrieval in a factor space needs the head applied to both
  sides first -- use :meth:`embed_content` / :meth:`embed_geometry` /
  :meth:`embed_semantic`.  The trunk-space primary endpoint needs nothing.
* Batch composition: the content and geometry terms only see multi-positive
  structure when duplicate videos/orientations co-occur in a batch, so this
  loss pairs with ``batch.mode: video_x_subject`` -- NOT ``distinct_video``.
* CAVEAT (blueprint v3, gate before any geometry claim): the geometry head is
  exposed to the two confounds the adversarial review flagged -- possible
  sequence-blocked orientation (audit A) and mirrored gaze patterns.  Its
  outputs must clear the ocular battery and the audit-A verdict before being
  reported as neural viewpoint coding.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    ContrastiveLoss,
    get_meta,
    mask_value,
    pairwise_eq,
    register_loss,
    safe_normalize,
    to_float,
)

_SPACES = ("exact", "content", "geometry", "semantic")


@register_loss("factorized")
class FactorizedFHMC(ContrastiveLoss):
    """Factorized hierarchical multi-positive contrastive objective.

    Parameters
    ----------
    dim
        Trunk embedding dimension (must equal the projector's output D).
    d_content, d_geometry, d_semantic
        Width of each factor head.  Geometry is deliberately narrow: it only
        has to carry a 4-way code.
    lambda_exact, lambda_content, lambda_geometry, lambda_semantic,
    lambda_disentangle
        Term weights.  Setting a lambda to 0 skips that term entirely.
    ce_weight
        Weight of the auxiliary 4-way orientation cross-entropy inside the
        geometry term (0 disables the classifier path).
    sigma
        Bandwidth of the affect kernel in z-score units (|dv|+|da|+|dt|).
    temp_* / learnable_temperature
        Per-space CLIP-style temperature, stored as log-scale and clamped to
        [min_scale, max_scale] like every other loss in this package.
    renormalize
        Re-normalize inputs defensively (contract says they arrive
        normalized; this guards collapsed rows).
    """

    def __init__(
        self,
        dim: int,
        d_content: int = 128,
        d_geometry: int = 32,
        d_semantic: int = 64,
        lambda_exact: float = 1.0,
        lambda_content: float = 1.0,
        lambda_geometry: float = 0.5,
        lambda_semantic: float = 0.25,
        lambda_disentangle: float = 0.1,
        ce_weight: float = 0.5,
        sigma: float = 1.0,
        temp_exact: float = 0.04,
        temp_content: float = 0.07,
        temp_geometry: float = 0.1,
        temp_semantic: float = 0.07,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.dim = int(dim)
        self.sigma = float(sigma)
        self.renormalize = bool(renormalize)
        self.ce_weight = float(ce_weight)
        self.lambdas = {
            "exact": float(lambda_exact),
            "content": float(lambda_content),
            "geometry": float(lambda_geometry),
            "semantic": float(lambda_semantic),
            "disent": float(lambda_disentangle),
        }

        # factor heads, shared across modalities (subspace interpretation)
        self.head_content = nn.Linear(dim, int(d_content), bias=False)
        self.head_geometry = nn.Linear(dim, int(d_geometry), bias=False)
        self.head_semantic = nn.Linear(dim, int(d_semantic), bias=False)
        # auxiliary orientation classifier on the EEG geometry embedding
        self.orient_head = nn.Linear(int(d_geometry), 4)

        # per-space temperatures (TemperatureMixin is single-temperature, so
        # the four log-scales are registered directly)
        self._temp_lo = math.log(float(min_scale))
        self._temp_hi = math.log(float(max_scale))
        temps = {
            "exact": temp_exact,
            "content": temp_content,
            "geometry": temp_geometry,
            "semantic": temp_semantic,
        }
        for space, t in temps.items():
            if t <= 0:
                raise ValueError(f"temp_{space} must be > 0, got {t}")
            init = torch.tensor(math.log(1.0 / float(t)), dtype=torch.float32)
            if not (self._temp_lo <= float(init) <= self._temp_hi):
                raise ValueError(
                    f"temp_{space}={t} implies scale {1.0 / t:.3f} outside "
                    f"[{min_scale}, {max_scale}]"
                )
            if learnable_temperature:
                setattr(self, f"logit_scale_{space}", nn.Parameter(init))
            else:
                self.register_buffer(f"logit_scale_{space}", init)

        self.requires_meta = (
            "condition_id",
            "video_id",
            "orientation",
            "valence",
            "arousal",
            "threat",
        )

    # ------------------------------------------------------------------ #
    # public helpers (eval wiring)
    # ------------------------------------------------------------------ #
    def scale(self, space: str) -> torch.Tensor:
        raw = getattr(self, f"logit_scale_{space}")
        return raw.clamp(min=self._temp_lo, max=self._temp_hi).exp()

    def embed_content(self, z: torch.Tensor) -> torch.Tensor:
        """Project trunk embeddings into the flip-invariant content space."""
        return safe_normalize(self.head_content(z))

    def embed_geometry(self, z: torch.Tensor) -> torch.Tensor:
        """Project trunk embeddings into the flip-equivariant geometry space."""
        return safe_normalize(self.head_geometry(z))

    def embed_semantic(self, z: torch.Tensor) -> torch.Tensor:
        """Project trunk embeddings into the affect-semantic space."""
        return safe_normalize(self.head_semantic(z))

    # ------------------------------------------------------------------ #
    # term machinery
    # ------------------------------------------------------------------ #
    def _multi_positive_nce(
        self,
        za: torch.Tensor,
        zb: torch.Tensor,
        pos: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric multi-positive InfoNCE.

        ``L_i = -log( sum_{j in P_i} exp(l_ij) / sum_j exp(l_ij) )`` averaged
        over rows that have at least one positive AND one negative, in both
        directions.  Returns (mean_loss_over_valid, n_valid_rows_tensor).
        """
        fill = None
        losses = []
        n_valid = za.new_zeros(())
        for logits, mask in (
            (scale * za @ zb.t(), pos),
            (scale * zb @ za.t(), pos.t()),
        ):
            if fill is None:
                fill = mask_value(logits.dtype)
            has_pos = mask.any(dim=1)
            has_neg = (~mask).any(dim=1)
            valid = has_pos & has_neg
            if not bool(valid.any()):
                continue
            pos_logits = logits.masked_fill(~mask, fill)
            lse_pos = torch.logsumexp(pos_logits, dim=1)
            lse_all = torch.logsumexp(logits, dim=1)
            losses.append((lse_all - lse_pos)[valid].mean())
            n_valid = n_valid + valid.sum()
        if not losses:
            return self._zero_loss(za, zb), n_valid
        return torch.stack(losses).mean(), n_valid

    def _semantic_term(
        self,
        zs_e: torch.Tensor,
        zs_v: torch.Tensor,
        affect_dist: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Soft-target cross-entropy against the affect kernel (both ways)."""
        target = F.softmax(-affect_dist / self.sigma, dim=1)
        losses = []
        for logits, tgt in (
            (scale * zs_e @ zs_v.t(), target),
            (scale * zs_v @ zs_e.t(), target.t()),
        ):
            logp = F.log_softmax(logits, dim=1)
            losses.append(-(tgt * logp).sum(dim=1).mean())
        return torch.stack(losses).mean()

    @staticmethod
    def _cross_covariance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mean squared entry of the batch cross-covariance matrix."""
        a_c = a - a.mean(dim=0, keepdim=True)
        b_c = b - b.mean(dim=0, keepdim=True)
        denom = max(a.shape[0] - 1, 1)
        cov = a_c.t() @ b_c / denom
        return (cov ** 2).mean()

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, z_eeg: torch.Tensor, z_vid: torch.Tensor, meta) -> Dict:
        self.check_meta(meta)
        if z_eeg.shape[1] != self.dim:
            raise ValueError(
                f"factorized loss was built with dim={self.dim} but received "
                f"embeddings of dim {z_eeg.shape[1]}; set loss.dim to the "
                f"projector output width"
            )
        if self.renormalize:
            z_eeg = safe_normalize(z_eeg)
            z_vid = safe_normalize(z_vid)
        B = z_eeg.shape[0]

        cond = get_meta(meta, "condition_id")
        vid = get_meta(meta, "video_id")
        orient = get_meta(meta, "orientation")
        affect = torch.stack(
            [
                get_meta(meta, "valence"),
                get_meta(meta, "arousal"),
                get_meta(meta, "threat"),
            ],
            dim=1,
        )
        # (B, B) L1 distance in affect space
        affect_dist = torch.cdist(affect, affect, p=1)

        logs: Dict[str, float] = {}
        terms: Dict[str, torch.Tensor] = {}
        total_valid = 0.0

        # --- exact (trunk space) --------------------------------------- #
        if self.lambdas["exact"] > 0:
            loss_e, nv = self._multi_positive_nce(
                z_eeg, z_vid, pairwise_eq(cond), self.scale("exact")
            )
            terms["exact"] = loss_e
            logs["exact/raw"] = to_float(loss_e)
            logs["exact/n_valid"] = to_float(nv)
            total_valid += to_float(nv)

        # --- content (flip-invariant) ---------------------------------- #
        zc_e = None
        if self.lambdas["content"] > 0:
            zc_e, zc_v = self.embed_content(z_eeg), self.embed_content(z_vid)
            loss_c, nv = self._multi_positive_nce(
                zc_e, zc_v, pairwise_eq(vid), self.scale("content")
            )
            terms["content"] = loss_c
            logs["content/raw"] = to_float(loss_c)
            logs["content/n_valid"] = to_float(nv)
            total_valid += to_float(nv)

        # --- geometry (flip-equivariant) ------------------------------- #
        zg_e = None
        if self.lambdas["geometry"] > 0:
            zg_e, zg_v = self.embed_geometry(z_eeg), self.embed_geometry(z_vid)
            loss_g, nv = self._multi_positive_nce(
                zg_e, zg_v, pairwise_eq(orient), self.scale("geometry")
            )
            if self.ce_weight > 0:
                ce = F.cross_entropy(self.orient_head(zg_e), orient)
                logs["geometry/ce"] = to_float(ce)
                with torch.no_grad():
                    acc = (
                        (self.orient_head(zg_e).argmax(1) == orient)
                        .float()
                        .mean()
                    )
                logs["geometry/orient_acc"] = to_float(acc)
                loss_g = loss_g + self.ce_weight * ce
            terms["geometry"] = loss_g
            logs["geometry/raw"] = to_float(loss_g)
            logs["geometry/n_valid"] = to_float(nv)
            total_valid += to_float(nv)

        # --- semantic (affect kernel) ---------------------------------- #
        if self.lambdas["semantic"] > 0 and B >= 2:
            zs_e, zs_v = self.embed_semantic(z_eeg), self.embed_semantic(z_vid)
            loss_s = self._semantic_term(
                zs_e, zs_v, affect_dist, self.scale("semantic")
            )
            terms["semantic"] = loss_s
            logs["semantic/raw"] = to_float(loss_s)

        # --- disentangle ------------------------------------------------ #
        if (
            self.lambdas["disent"] > 0
            and zc_e is not None
            and zg_e is not None
            and B >= 2
        ):
            loss_d = self._cross_covariance(zc_e, zg_e)
            terms["disent"] = loss_d
            logs["disent/raw"] = to_float(loss_d)

        if not terms:
            total = self._zero_loss(z_eeg, z_vid)
        else:
            total = torch.stack(
                [self.lambdas[k] * v for k, v in terms.items()]
            ).sum()

        logs["n_valid"] = total_valid
        logs["degenerate"] = 0.0 if total_valid > 0 else 1.0
        for space in _SPACES:
            logs[f"scale/{space}"] = to_float(self.scale(space))
        return {"loss": total, "logs": logs}


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    for tag, kw in [
        ("random", {}),
        ("many_dups", {"n_unique_conditions": 4}),
        ("single_condition", {"single_condition": True}),
        ("batch_of_1", {"batch_size": 1}),
    ]:
        fn = FactorizedFHMC(dim=256)
        z_eeg, z_vid, meta = make_dummy_batch(seed=0, **kw)
        out = fn(z_eeg, z_vid, meta)
        out["loss"].backward()
        head = {k: round(v, 3) for k, v in list(out["logs"].items())[:5]}
        print(f"{tag:18s} loss={out['loss'].item():8.4f} logs={head}")
    print("factorized self-test OK")
