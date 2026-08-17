"""LaBraM (Jiang et al., ICLR 2024) EEG foundation model, fine-tuned -- ladder rung 5.

INDICATIVE BASELINE ONLY (MISSION 5.3).  LaBraM was pretrained on ~2500 h of
continuous EEG chopped into **1-second, 200-sample patches at 200 Hz**.  Our epochs
are 120 samples (0-600 ms, contract B).  That interface mismatch is the whole story
of this rung and it is documented here rather than hidden.

The 200 -> 120 patch trap (measured, not guessed)
-------------------------------------------------
``braindecode.models.Labram`` accepts ``n_times=120`` with only a ``UserWarning``::

    UserWarning: patch_size (200) > n_times (120); setting patch_size = 120.

That warning under-states the damage by four orders of magnitude.  The neural
tokenizer is ``_SegmentPatch -> _TemporalConv``; ``_TemporalConv`` is a fully
convolutional stack (k=(1,15) s=(1,8) p=(0,7), then two k=(1,3) convs) whose *output
width* is ``8 * ((patch_size - 1)//8 + 1)``.  That width **is** the transformer's
``embed_dim``.  So

* ``patch_size = 200``  ->  ``embed_dim = 8 * 25 = 200``  (the pretrained width)
* ``patch_size = 120``  ->  ``embed_dim = 8 * 15 = 120``  (a different model)

and every tensor downstream of the patcher -- ``cls_token``, ``position_embedding``,
``temporal_embedding``, all 12 attention blocks, every LayerNorm -- changes shape.
Measured against ``braindecode/labram-pretrained``:

===============  =========================  ==============================
build            checkpoint tensors usable  checkpoint parameters usable
===============  =========================  ==============================
``n_times=120``  12 / 221                   576 / 2,108,112  = **0.03 %**
``n_times=200``  221 / 221                  5,817,136 / 5,817,136 = **100 %**
===============  =========================  ==============================

The 12 tensors that survive at 120 are exactly the three ``_TemporalConv`` convs and
their GroupNorms -- the only shape-independent layers in the model.  A LaBraM built at
``n_times=120`` is therefore **not a fine-tuned foundation model, it is a randomly
initialised 2.1 M-parameter transformer wearing LaBraM's name**.  This module refuses
to build that object unless ``allow_random_init=True`` is passed explicitly.

The window / padding decision (pre-registered)
---------------------------------------------
We keep the real 200-sample patcher and pad the epoch, because the alternative
destroys 99.97 % of the pretrained weights and would make the rung meaningless.

* PRIMARY: ``w0600`` (120 samples) padded to 200  ->  **padding fraction 0.40**.
  This keeps the rung comparable to every other baseline on the ladder, which all
  read the same 0-600 ms window.
* SENSITIVITY: ``wm100_800`` (180 samples) padded to 200 -> padding fraction 0.10,
  but that window is baseline-contaminated by design (contract B).  Run it *labelled*,
  never as the headline.

Padding is ``replicate`` + ``center`` by default: the epoch sits in the middle of the
patch, the 40 samples before onset hold the value at 0 ms and the 40 after hold the
value at 600 ms.  Replicate is C0-continuous, so it fabricates no broadband transient
(zero-padding does, on data that carry no baseline correction and therefore no
guarantee of a zero mean), and it duplicates no evoked event (``reflect`` does).
``pad_mode`` / ``pad_align`` expose the alternatives; the realised fraction is
reported by :meth:`describe` under ``pad_fraction`` so it lands in the run log.

Channel mapping is free: all 64 BioSemi-64 names in ``derived/channels.json`` are
present in ``LABRAM_CHANNEL_ORDER`` (verified 64/64), so the per-channel position
embeddings are selected by name via ``Labram.forward(ch_names=...)`` with no
interpolation and no ``InterpolatedLaBraM``.

Honest accounting of what is and is not pretrained
--------------------------------------------------
* backbone (5.82 M params): **100 % from the checkpoint**.  The single tensor that
  cannot be copied verbatim, ``temporal_embedding``, is *sliced* from the pretrained
  ``(1, 16, 200)`` to ``(1, 2, 200)`` -- the patch-position embeddings for the patches
  we actually have -- so its values are pretrained too, not random.
* projector + subject conditioner: random, as on every other rung.
* amplitude convention: LaBraM was pretrained on microvolts / 100.  Our pipeline feeds
  robust-scaled units (measured on 512 real ``w0600`` epochs: mean -0.002, std 0.87,
  p99 |x| 2.84).  ``input_scale`` exposes the knob; sweeping {3, 1, 0.3, 0.1, 0.03,
  0.01} on those trials gave the widest feature spread and the highest effective rank
  at 1.0 (mean |cos| 0.27 / eff-rank 8.5, vs 0.34 / 5.1 at 0.1), so the default is
  measured rather than assumed -- but it is *not* a calibration to LaBraM's units.

Fine-tuning knobs: ``freeze_backbone``, ``n_unfrozen_blocks`` (partial unfreeze of the
top blocks), and ``backbone_lr_scale`` consumed by :meth:`param_groups` -- see the note
on that method, the shipped trainer does not call it yet.
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..heads import ResidualProjection
from .base import EEGEncoder, count_parameters, register_eeg_encoder

__all__ = ["LabramEncoder", "BIOSEMI64_CHANNEL_ORDER", "load_pretrained_labram"]

log = logging.getLogger(__name__)


#: Default Hugging Face repo carrying the LaBraM-base weights.
LABRAM_HF_REPO = "braindecode/labram-pretrained"

#: Canonical BioSemi-64 order (contract B, ``derived/channels.json``).  Kept here so the
#: encoder can be built without the data root mounted; :func:`_default_ch_names`
#: cross-checks it against ``channels.json`` whenever that file is reachable and raises
#: on disagreement, because a silently permuted montage is a silently wrong topography.
BIOSEMI64_CHANNEL_ORDER: Tuple[str, ...] = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)

#: Architecture keys copied verbatim from the checkpoint's hub config.  Deliberately
#: excludes the class-valued keys (``qk_norm``, ``norm_layer``, ``activation``), whose
#: braindecode defaults already match the checkpoint -- and any drift there would be
#: caught by the 100 %-coverage assertion in :func:`load_pretrained_labram` anyway.
_ARCH_KEYS: Tuple[str, ...] = (
    "patch_size", "learned_patcher", "embed_dim", "conv_in_channels", "conv_out_channels",
    "num_layers", "num_heads", "mlp_ratio", "qkv_bias", "init_values", "use_abs_pos_emb",
    "use_mean_pooling", "init_scale", "neural_tokenizer", "attn_head_dim",
)

_POOL_MODES = ("cls", "mean", "cls_mean", "flatten")
_PAD_MODES = ("replicate", "zeros", "reflect")
_PAD_ALIGNS = ("center", "left", "right")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _import_labram():
    """Import braindecode's LaBraM lazily, with an actionable error message."""
    try:
        from braindecode.models import Labram  # noqa: PLC0415
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "the 'labram' encoder needs braindecode>=1.7 with Hugging Face Hub support "
            "(`pip install 'braindecode[hug]'`); do NOT let pip move torch/numpy/mne/"
            f"torchaudio while doing it. Original error: {exc}"
        ) from exc
    return Labram, LABRAM_CHANNEL_ORDER


def _default_ch_names(n_channels: int) -> List[str]:
    """The 64 BioSemi channel names in acquisition order, cross-checked against disk."""
    if n_channels != len(BIOSEMI64_CHANNEL_ORDER):
        raise ValueError(
            f"labram needs explicit ch_names for n_channels={n_channels}; only the "
            f"{len(BIOSEMI64_CHANNEL_ORDER)}-channel BioSemi montage has a built-in order"
        )
    names = list(BIOSEMI64_CHANNEL_ORDER)
    root = os.environ.get("TACTUS_DATA_ROOT")
    if root:
        path = Path(root) / "derived" / "channels.json"
        try:
            blob = json.loads(path.read_text())
        except Exception:  # file absent / unreadable: fall back to the constant
            return names
        on_disk = blob.get("channels") if isinstance(blob, dict) else blob
        if isinstance(on_disk, list) and len(on_disk) == len(names):
            if [c.upper() for c in on_disk] != [c.upper() for c in names]:
                raise RuntimeError(
                    f"{path} disagrees with BIOSEMI64_CHANNEL_ORDER in fm_labram.py. "
                    "The channel order decides which LaBraM position embedding each "
                    "electrode reads, so this must not be papered over -- pass "
                    "ch_names=<the on-disk order> explicitly or fix the constant."
                )
            return [str(c) for c in on_disk]
    return names


def load_pretrained_labram(
    backbone: nn.Module,
    repo_id: str = LABRAM_HF_REPO,
    *,
    local_files_only: Optional[bool] = None,
    min_fraction: float = 0.999,
) -> Dict[str, Any]:
    """Transplant ``repo_id``'s weights into ``backbone`` and *prove* it happened.

    ``Labram.from_pretrained(repo, n_times=200, n_chans=64)`` does not work: the hub
    config pins ``chs_info`` to the canonical 128 channels (so ``n_chans=64`` raises a
    conflict) and ``temporal_embedding`` is shaped by ``n_times`` (so a 200-sample model
    hits ``size mismatch ... [1, 16, 200] vs [1, 2, 200]``).  We therefore materialise
    the checkpoint at its native geometry and copy tensor by tensor, slicing the leading
    ``n_patches + 1`` rows of ``temporal_embedding``.

    Raises if fewer than ``min_fraction`` of the backbone's parameters came from the
    checkpoint -- a fine-tuning baseline that quietly trained from scratch is worse than
    no baseline at all.
    """
    Labram, _ = _import_labram()

    def _fetch(**kw):
        return Labram.from_pretrained(repo_id, **kw)

    if local_files_only is None:
        try:
            src = _fetch()
        except Exception as exc:  # no network on a compute node: use the HF_HOME cache
            try:
                src = _fetch(local_files_only=True)
            except Exception:
                raise RuntimeError(
                    f"could not load pretrained LaBraM from {repo_id!r}. Populate the "
                    f"cache from a login node with HF_HOME={os.environ.get('HF_HOME')} "
                    f"set. Original error: {exc}"
                ) from exc
    else:
        src = _fetch(local_files_only=bool(local_files_only))

    src_sd = src.state_dict()
    tgt_sd = backbone.state_dict()
    staged: Dict[str, torch.Tensor] = {}
    sliced: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for key, want in tgt_sd.items():
        have = src_sd.get(key)
        if have is None:
            skipped.append((key, "absent from checkpoint"))
            continue
        if have.shape == want.shape:
            staged[key] = have.detach().clone()
        elif have.dim() == want.dim() and all(a >= b for a, b in zip(have.shape, want.shape)):
            staged[key] = have[tuple(slice(0, b) for b in want.shape)].detach().clone()
            sliced.append(f"{key}: {tuple(have.shape)} -> {tuple(want.shape)}")
        else:
            skipped.append((key, f"{tuple(have.shape)} vs {tuple(want.shape)}"))

    missing, unexpected = backbone.load_state_dict(staged, strict=False)
    live = backbone.state_dict()
    not_applied = [k for k in staged if not torch.equal(live[k], staged[k])]
    if not_applied:  # pragma: no cover - would mean torch lied to us
        raise RuntimeError(f"load_state_dict silently dropped {len(not_applied)} tensors")

    n_total = sum(t.numel() for t in tgt_sd.values())
    n_loaded = sum(tgt_sd[k].numel() for k in staged)
    frac = n_loaded / max(1, n_total)
    report = {
        "repo_id": repo_id,
        "tensors_total": len(tgt_sd),
        "tensors_from_checkpoint": len(staged),
        "tensors_sliced": sliced,
        "tensors_skipped": skipped,
        "params_total": n_total,
        "params_from_checkpoint": n_loaded,
        "fraction": round(frac, 6),
        "load_state_dict_missing": list(missing),
        "load_state_dict_unexpected": list(unexpected),
    }
    if frac < float(min_fraction):
        raise RuntimeError(
            f"only {frac:.4%} of the LaBraM backbone came from {repo_id!r} "
            f"({n_loaded:,}/{n_total:,} params; skipped={skipped[:6]}). This is a "
            "randomly initialised transformer, not a foundation model. Fix the "
            "geometry (patch_size/n_times) or pass pretrained=False deliberately."
        )
    del src
    return report


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------


@register_eeg_encoder("labram", "fm_labram", "labram_ft")
class LabramEncoder(EEGEncoder):
    """Fine-tuned LaBraM-base trunk + the standard TACTUS projection head (contract F).

    Parameters
    ----------
    pretrained:
        Load ``hf_repo`` into the backbone.  ``False`` is only legal together with
        ``allow_random_init=True`` -- see :func:`load_pretrained_labram`.
    hf_repo, local_files_only:
        Checkpoint location; ``local_files_only=True`` forces the ``HF_HOME`` cache.
    patch_size:
        ``None`` (default) takes the checkpoint's value (200).  Passing a different one
        while ``pretrained=True`` raises, because it silently rebuilds the whole
        transformer at a different ``embed_dim`` (see the module docstring).
    pad_mode, pad_align:
        How the ``n_times``-sample epoch is padded up to a whole number of patches.
        ``"replicate"``/``"center"`` by default.
    ch_names:
        Channel names in the order of the data's channel axis; defaults to the BioSemi-64
        order cross-checked against ``derived/channels.json``.  Names are matched
        case-insensitively against ``LABRAM_CHANNEL_ORDER`` to pick position embeddings.
    pool:
        ``"cls"`` (200-d), ``"mean"`` (200-d, mean over channel tokens), ``"cls_mean"``
        (400-d, default) or ``"flatten"`` (``n_chans * n_patches * 200``; keeps the full
        topography but adds a large random projector).
    input_scale:
        Multiplies the input.  LaBraM pretrained on microvolts/100; we feed robust-scaled
        units.  1.0 is a choice, not a calibration.
    freeze_backbone, n_unfrozen_blocks:
        ``freeze_backbone=True`` freezes the whole trunk (linear probe on the frozen
        foundation model).  ``n_unfrozen_blocks=k`` freezes the trunk and then re-enables
        the top ``k`` of 12 attention blocks plus the final norm -- a partial fine-tune;
        it takes effect whether or not ``freeze_backbone`` is set, since "k blocks are
        unfrozen" already implies the rest are not.  ``None`` (default) leaves the whole
        trunk trainable.
    backbone_lr_scale:
        Multiplier for the backbone's learning rate, consumed by :meth:`param_groups`.
    drop_prob, attn_drop_prob, drop_path_prob:
        Regularisation.  ``drop_path_prob=0.1`` is LaBraM's own fine-tuning default and
        changes no parameter shapes.
    allow_random_init:
        Escape hatch for an ablation that deliberately wants the architecture without the
        weights.  Emits a loud warning; never use it for the reported rung.
    proj_hidden, proj_blocks, proj_dropout:
        Residual projector geometry, as in :class:`~tactus.models.eeg.tsconv.TSConvEncoder`.
    **base:
        :class:`~tactus.models.eeg.base.EEGEncoder` arguments.
    """

    config_aliases = {
        "hf_repo": ("repo_id", "checkpoint", "weights"),
        "proj_hidden": ("d_model", "d_hidden"),
        "proj_dropout": ("dropout", "drop"),
        "n_unfrozen_blocks": ("unfreeze_last_n", "n_trainable_blocks"),
        "backbone_lr_scale": ("backbone_lr_mult", "lr_scale_backbone"),
    }

    def __init__(
        self,
        *,
        pretrained: bool = True,
        hf_repo: str = LABRAM_HF_REPO,
        local_files_only: Optional[bool] = None,
        patch_size: Optional[int] = None,
        pad_mode: str = "replicate",
        pad_align: str = "center",
        ch_names: Optional[Sequence[str]] = None,
        pool: str = "cls_mean",
        input_scale: float = 1.0,
        freeze_backbone: bool = False,
        n_unfrozen_blocks: Optional[int] = None,
        backbone_lr_scale: float = 0.1,
        drop_prob: float = 0.0,
        attn_drop_prob: float = 0.0,
        drop_path_prob: float = 0.1,
        allow_random_init: bool = False,
        min_pretrained_fraction: float = 0.999,
        proj_hidden: Optional[int] = None,
        proj_blocks: int = 1,
        proj_dropout: float = 0.5,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        Labram, labram_order = _import_labram()

        if pool not in _POOL_MODES:
            raise ValueError(f"pool must be one of {_POOL_MODES}, got {pool!r}")
        if pad_mode not in _PAD_MODES:
            raise ValueError(f"pad_mode must be one of {_PAD_MODES}, got {pad_mode!r}")
        if pad_align not in _PAD_ALIGNS:
            raise ValueError(f"pad_align must be one of {_PAD_ALIGNS}, got {pad_align!r}")
        self.pool = str(pool)
        self.pad_mode = str(pad_mode)
        self.pad_align = str(pad_align)
        self.input_scale = float(input_scale)
        self.backbone_lr_scale = float(backbone_lr_scale)
        self.hf_repo = str(hf_repo)

        # ---- channels -----------------------------------------------------------------
        names = [str(c) for c in ch_names] if ch_names is not None else _default_ch_names(self.n_channels)
        if len(names) != self.n_channels:
            raise ValueError(f"ch_names has {len(names)} entries but n_channels={self.n_channels}")
        known = {c.upper() for c in labram_order}
        unknown = [c for c in names if c.upper() not in known]
        if unknown:
            raise ValueError(
                f"channels absent from LABRAM_CHANNEL_ORDER: {unknown}. LaBraM indexes its "
                "position embeddings by electrode name; use InterpolatedLaBraM for a "
                "montage it has never seen."
            )
        self.ch_names: List[str] = names

        # ---- architecture, taken from the checkpoint ------------------------------------
        arch: Dict[str, Any] = {}
        if pretrained:
            arch = self._checkpoint_arch(Labram, hf_repo, local_files_only)
        ckpt_patch = int(arch.get("patch_size", 200) or 200)
        if patch_size is None:
            patch_size = ckpt_patch
        elif pretrained and int(patch_size) != ckpt_patch:
            raise ValueError(
                f"patch_size={patch_size} != the checkpoint's {ckpt_patch}. LaBraM's "
                f"transformer width is 8 * ceil(patch_size / 8) = the patcher's output "
                f"width, so changing it discards essentially the entire checkpoint "
                f"(measured: 0.03 % of parameters survive at patch_size=120). Pad the "
                f"epoch instead -- that is what pad_mode/pad_align are for."
            )
        self.patch_size = int(patch_size)

        # Pad the window up to a whole number of patches; the backbone is built for that
        # length and every call to extract_features (including TimeWindowHeads' short
        # sub-windows) is padded to exactly it.
        self.n_patches = max(1, math.ceil(self.n_times / self.patch_size))
        self.backbone_n_times = self.n_patches * self.patch_size
        self.pad_total = self.backbone_n_times - self.n_times
        self.pad_fraction = self.pad_total / self.backbone_n_times

        kwargs = {k: arch[k] for k in _ARCH_KEYS if k in arch and arch[k] is not None}
        kwargs.pop("patch_size", None)
        kwargs.update(
            n_times=self.backbone_n_times,
            n_chans=self.n_channels,
            n_outputs=0,          # no classifier head: we read tokens, not logits
            sfreq=200.0,
            chs_info=None,        # position embeddings are selected per batch by name
            patch_size=self.patch_size,
            drop_prob=float(drop_prob),
            attn_drop_prob=float(attn_drop_prob),
            drop_path_prob=float(drop_path_prob),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.backbone = Labram(**kwargs)
        # braindecode downgrades the geometry trap to a UserWarning; we promote it back.
        trap = [str(w.message) for w in caught if "patch_size" in str(w.message)]
        if trap or int(self.backbone.patch_size) != self.patch_size:
            why = "; ".join(trap) or f"patch_size {self.patch_size} -> {self.backbone.patch_size}"
            raise RuntimeError(
                f"braindecode rewrote the LaBraM geometry ({why}). That silently changes "
                "embed_dim and discards essentially the whole checkpoint; refusing to build."
            )
        for w in caught:  # anything else stays visible
            warnings.warn(w.message, w.category, stacklevel=2)

        # ---- weights --------------------------------------------------------------------
        if pretrained:
            self.pretrained_report = load_pretrained_labram(
                self.backbone,
                hf_repo,
                local_files_only=local_files_only,
                min_fraction=float(min_pretrained_fraction),
            )
        elif allow_random_init:
            warnings.warn(
                "LabramEncoder built with pretrained=False: this is a randomly initialised "
                "transformer and is NOT a foundation-model baseline. Label any result from "
                "it accordingly.",
                UserWarning,
                stacklevel=2,
            )
            self.pretrained_report = {"repo_id": None, "fraction": 0.0, "random_init": True}
        else:
            raise ValueError(
                "pretrained=False without allow_random_init=True. A randomly initialised "
                "'foundation model' is worthless as a baseline (MISSION 5.3); if you really "
                "want the architecture-only control, ask for it explicitly."
            )

        # ---- freezing --------------------------------------------------------------------
        self.freeze_backbone = bool(freeze_backbone)
        self.n_unfrozen_blocks = None if n_unfrozen_blocks is None else int(n_unfrozen_blocks)
        self._apply_freezing()

        self.proj_hidden = proj_hidden
        self.proj_blocks = int(proj_blocks)
        self.proj_dropout = float(proj_dropout)
        self.setup_head()

        # The trainer does not call describe(), so the interface decision is logged here:
        # the padding fraction and the pretrained fraction must appear in every run's log.
        log.info(
            "LaBraM: backbone %.2fM params, %.1f%% from %s (%s tensors) | epoch %d samples "
            "-> %d x patch %d = %d samples, %d padded (%.0f%% %s/%s) | pool=%s dim=%d | "
            "trainable backbone %.2fM | INDICATIVE baseline: patch/epoch mismatch is real",
            count_parameters(self.backbone, trainable_only=False) / 1e6,
            100.0 * float(self.pretrained_report.get("fraction", 0.0)),
            self.pretrained_report.get("repo_id"),
            f"{self.pretrained_report.get('tensors_from_checkpoint')}/"
            f"{self.pretrained_report.get('tensors_total')}",
            self.n_times, self.n_patches, self.patch_size, self.backbone_n_times,
            self.pad_total, 100.0 * self.pad_fraction, self.pad_mode, self.pad_align,
            self.pool, self.feature_dim,
            count_parameters(self.backbone, trainable_only=True) / 1e6,
        )

    # ---------------------------------------------------------------------------------
    # construction helpers
    # ---------------------------------------------------------------------------------

    @staticmethod
    def _checkpoint_arch(Labram, repo_id: str, local_files_only: Optional[bool]) -> Dict[str, Any]:
        """Read the checkpoint's ``config.json`` so our trunk is shape-identical to it."""
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            kw: Dict[str, Any] = {}
            if local_files_only is not None:
                kw["local_files_only"] = bool(local_files_only)
            path = hf_hub_download(repo_id=repo_id, filename="config.json", **kw)
            with open(path, "r") as fh:
                return dict(json.load(fh))
        except Exception:
            # Not fatal: braindecode's defaults already match LaBraM-base, and
            # load_pretrained_labram's 100 %-coverage assertion is the real guard.
            return {}

    def _apply_freezing(self) -> None:
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if self.n_unfrozen_blocks is not None:
            k = self.n_unfrozen_blocks
            blocks = list(self.backbone.blocks)
            if not 0 <= k <= len(blocks):
                raise ValueError(f"n_unfrozen_blocks must be in [0, {len(blocks)}], got {k}")
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for blk in blocks[len(blocks) - k:] if k else []:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for mod in (self.backbone.norm, self.backbone.fc_norm):
                if isinstance(mod, nn.Module):
                    for p in mod.parameters():
                        p.requires_grad_(True)

    # ---------------------------------------------------------------------------------
    # trunk
    # ---------------------------------------------------------------------------------

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad ``(B, C, T)`` up to :attr:`backbone_n_times`."""
        t = int(x.shape[-1])
        target = self.backbone_n_times
        if t == target:
            return x
        if t > target:
            raise ValueError(
                f"LabramEncoder was built for at most {target} samples "
                f"({self.n_patches} x patch_size {self.patch_size}) but got {t}. Rebuild "
                f"the encoder with a larger n_times."
            )
        total = target - t
        if self.pad_align == "center":
            left = total // 2
        elif self.pad_align == "left":  # data first, padding after
            left = 0
        else:  # "right": padding first, data last
            left = total
        right = total - left
        mode = self.pad_mode
        if mode == "reflect" and max(left, right) >= t:
            # torch requires pad < input length for reflect.  Record the
            # downgrade instead of silently reporting a padding mode that was
            # not used -- describe() is the provenance of this rung's number.
            mode = "replicate"
            if getattr(self, "_pad_mode_effective", None) != mode:
                self._pad_mode_effective = mode
                log.warning(
                    "%s: pad_mode='reflect' is impossible for a %d-sample input "
                    "padded by %d (torch requires pad < input length); using "
                    "'replicate'. describe()['pad_mode_effective'] records this.",
                    type(self).__name__, t, max(left, right),
                )
        if mode == "zeros":
            return F.pad(x, (left, right), mode="constant", value=0.0)
        return F.pad(x, (left, right), mode=mode)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self.input_scale != 1.0:
            x = x * self.input_scale
        x = self._pad(x)
        out = self.backbone(x, return_features=True, ch_names=self.ch_names)
        tokens, cls = out["features"], out["cls_token"]  # (B, C*P, E), (B, E)
        if self.pool == "cls":
            return cls
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "cls_mean":
            return torch.cat([cls, tokens.mean(dim=1)], dim=-1)
        return tokens.flatten(1)

    def build_projector(self, in_dim: int) -> nn.Module:  # noqa: D102
        return ResidualProjection(
            in_dim,
            self.embed_dim,
            hidden_dim=self.proj_hidden if self.proj_hidden is not None else self.embed_dim,
            n_blocks=self.proj_blocks,
            dropout=self.proj_dropout,
            norm="layernorm",
            normalize=False,  # the base class L2-normalizes once, at the very end
        )

    # ---------------------------------------------------------------------------------
    # fine-tuning support
    # ---------------------------------------------------------------------------------

    def param_groups(self, base_lr: float, weight_decay: float = 0.05) -> List[Dict[str, Any]]:
        """Optimizer groups giving the pretrained trunk ``backbone_lr_scale * base_lr``.

        .. note::
           ``tactus/train/trainer.py`` currently builds its groups directly from
           ``encoder.named_parameters()`` with a single global LR, so this method is
           called by Trainer._build, which prepends the returned groups, so the
           backbone really does train at ``backbone_lr_scale`` x the projector LR
           (``if hasattr(self.encoder, "param_groups"): groups = ...``).  Until then the
           usable knobs are ``freeze_backbone`` / ``n_unfrozen_blocks``, and a gradient
           hook would NOT be a substitute -- AdamW normalises by the gradient's own
           second moment, so scaling gradients does not scale the step.
        """
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        buckets: Dict[Tuple[bool, bool], List[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            key = (id(p) in backbone_ids, p.ndim <= 1 or name.endswith(".bias"))
            buckets.setdefault(key, []).append(p)
        groups: List[Dict[str, Any]] = []
        for (is_backbone, no_decay), params in buckets.items():
            groups.append(
                {
                    "params": params,
                    "lr": base_lr * (self.backbone_lr_scale if is_backbone else 1.0),
                    "weight_decay": 0.0 if no_decay else weight_decay,
                    "name": ("backbone" if is_backbone else "head") + ("_no_decay" if no_decay else ""),
                }
            )
        return groups

    # ---------------------------------------------------------------------------------
    # reporting
    # ---------------------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:  # noqa: D102
        out = super().describe()
        rep = dict(getattr(self, "pretrained_report", {}))
        out.update(
            {
                "backbone": "braindecode.models.Labram (LaBraM-base)",
                "pretrained_repo": rep.get("repo_id"),
                "pretrained_param_fraction": rep.get("fraction"),
                "pretrained_tensors": (
                    f"{rep.get('tensors_from_checkpoint')}/{rep.get('tensors_total')}"
                ),
                "pretrained_tensors_sliced": rep.get("tensors_sliced"),
                "backbone_parameters": count_parameters(self.backbone, trainable_only=False),
                "backbone_trainable": count_parameters(self.backbone, trainable_only=True),
                "patch_size": self.patch_size,
                "n_patches": self.n_patches,
                "epoch_samples": self.n_times,
                "backbone_samples": self.backbone_n_times,
                "pad_samples": self.pad_total,
                "pad_fraction": round(self.pad_fraction, 4),
                # reflect is impossible when the pad meets or exceeds the input
                # length (torch's constraint), which is the case for every
                # TimeWindowHeads sub-window; _pad falls back to replicate and
                # records it here so the provenance is not a fiction.
                "pad_mode_effective": getattr(self, "_pad_mode_effective", self.pad_mode),
                "pad_mode": self.pad_mode,
                "pad_align": self.pad_align,
                "pool": self.pool,
                "input_scale": self.input_scale,
                "freeze_backbone": self.freeze_backbone,
                "n_unfrozen_blocks": self.n_unfrozen_blocks,
                "backbone_lr_scale": self.backbone_lr_scale,
                "interface_note": (
                    f"LaBraM patch = {self.patch_size} samples (1 s @ 200 Hz); the epoch is "
                    f"{self.n_times} samples, so {self.pad_fraction:.0%} of every patch is "
                    f"{self.pad_mode} padding ({self.pad_align}-aligned). Building at "
                    f"n_times={self.n_times} instead would shrink patch_size and leave only "
                    f"0.03% of the checkpoint loadable."
                ),
            }
        )
        return out


if __name__ == "__main__":  # pragma: no cover - CPU smoke check for the server-side run
    import argparse

    ap = argparse.ArgumentParser(description="LaBraM encoder smoke test (CPU)")
    ap.add_argument("--n-times", type=int, default=120)
    ap.add_argument("--pool", default="cls_mean")
    args = ap.parse_args()

    for cond in ("none", "subject_token", "subject_layer", "sulora"):
        enc = LabramEncoder(n_times=args.n_times, embed_dim=256, pool=args.pool, subject_cond=cond).eval()
        x = torch.randn(4, 64, args.n_times)
        sid = torch.tensor([1, 2, 3, -1])
        with torch.no_grad():
            z = enc(x, sid)
        d = enc.describe()
        print(
            f"{cond:>14s}  params={count_parameters(enc):>10,d}  out={tuple(z.shape)}  "
            f"norm={z.norm(dim=-1).mean():.4f}  pretrained={d['pretrained_param_fraction']}  "
            f"pad={d['pad_fraction']}  rule={enc.unseen_subject_state()['rule']}"
        )
