"""EEGPT (Wang et al., NeurIPS 2024) EEG foundation model, fine-tuned -- ladder rung 5.

INDICATIVE BASELINE ONLY (MISSION 5.3, 3 folds).  Unlike the other two foundation
models on this rung, EEGPT is a *good* geometric fit to our epochs, and the
interface costs are small but non-zero.  They are enumerated here rather than
hidden.

Why EEGPT is the cleanest FM number in this ladder
--------------------------------------------------
EEGPT tokenises each channel independently into ``patch_size = 64`` sample patches
with ``patch_stride = 32``.  At our 200 Hz that is a 320 ms patch every 160 ms, so a
120-sample epoch (``w0600``, contract B) yields **3 patches with only 8 samples of
padding** -- 6.25 %, against 40 % for LaBraM and CBraMod, both of which need a
1-second (200-sample) patch.  Nothing in the checkpoint has to be resized, so

    **100.000 % of the 25,287,168 backbone parameters come from
    ``braindecode/eegpt-pretrained``** (103 / 103 tensors, verified tensor by tensor
    by :func:`load_pretrained_eegpt`, which raises if the fraction drops).

Three interface facts that are *not* free, in decreasing order of importance
---------------------------------------------------------------------------

**1. Sampling rate: 250 Hz pretraining vs 200 Hz data.**  The checkpoint's own
``config.json`` declares ``sfreq: 250`` (and ``n_times: 1000``, ``n_chans: 62``), where
``patch_size=64`` spans 256 ms.  Feeding 200 Hz
samples to the same convolution stretches every patch to 320 ms, i.e. the model sees
our signal 25 % *slower* than anything it was trained on -- a 10 Hz alpha rhythm
arrives looking like 8 Hz.  Two honest options, both implemented:

* ``resample_sfreq=None`` (default): feed the native 200 Hz samples.  No
  interpolation, exactly the samples every other rung reads, but the 1.25x temporal
  dilation above is real and uncorrected.
* ``resample_sfreq=250``: linearly upsample the epoch 120 -> 150 samples *inside* the
  encoder (600 ms of data either way -- the window is unchanged, only its
  representation).  Upsampling cannot alias, so this is a lossless re-representation
  that restores EEGPT's physical patch duration; the cost is 4 patches instead of 3
  and an interpolation the other rungs do not have.

The default is native; :meth:`describe` reports ``resample_sfreq``, ``model_sfreq``,
``patch_ms`` and ``pretrain_patch_ms`` so the log always says which one ran.

**2. Channel set: 64 BioSemi electrodes vs EEGPT's canonical 62.**  EEGPT indexes a
62-row channel embedding by electrode name (``EEGPT_CHANNELS``, a 10-20/10-10 subset
of MNE's ``standard_1020``).  Against ``derived/channels.json``:

* 60 of our 64 electrodes are in EEGPT's set and are passed through **verbatim**;
* 4 are not -- ``P9``, ``P10``, ``Iz``, ``AFz`` -- EEGPT has no embedding for them;
* 2 EEGPT channels are absent from BioSemi-64 -- ``PO5``, ``PO6``.

``channel_mode="interpolate"`` (default) uses ``braindecode.models.InterpolatedEEGPT``
with ``interpolation_mode="name_match"``: the 60 matching rows of the 62x64
projection matrix are one-hot (bit-exact copies, *no* smoothing of the channels EEGPT
knows), and only ``PO5``/``PO6`` are filled by a frozen MNE spherical-spline
interpolation over all 64 electrodes.  The four unmatched electrodes therefore enter
the model only through those two rows, with total absolute weight 0.014 (P9), 0.016
(P10), 0.036 (Iz), 0.003 (AFz) -- i.e. **P9, P10, Iz and AFz are effectively dropped**.
That is the cross-dataset channel-mapping cost this rung pays, and it is the number
MISSION section 9 asks for before the architecture is frozen.

Two alternatives are implemented for that decision:

* ``channel_mode="subset"``  -- feed only the 60 name-matched electrodes, no
  interpolation at all, ``PO5``/``PO6`` simply absent (the checkpoint still loads
  100 %, because ``chan_embed`` is sized ``max(62, n_chans)``).  Same four electrodes
  dropped, but nothing is fabricated.
* ``channel_mode="conv19"`` -- braindecode's *default*: a randomly initialised
  ``Conv1dWithConstraint(64 -> 19)`` onto the 19-channel 10-20 montage EEGPT's own
  linear probe uses.  Keeps all 64 inputs but throws away the topography the
  pretrained channel embeddings encode and adds 1,235 random parameters.  Not
  recommended; provided because it is what an unconfigured
  ``EEGPT.from_pretrained`` would silently do.

**3. Patch-grid padding.**  ``(T - 64) % 32`` must be 0.  120 -> 128 (8 samples,
6.25 %), 180 -> 192 (12 samples, 6.25 %), 150 (resampled) -> 160 (10 samples, 6.25 %).
braindecode pads this itself with **zeros** (``nn.ConstantPad1d(..., 0.0)``); our
epochs carry no baseline correction, so ``|x|`` at the epoch edge averages 0.53 in
robust-scaled units (~0.6 sigma) and a zero pad would fabricate a step transient in
the last patch.  This module therefore pads *before* the backbone with
``replicate``/left-align by default (data first, the 600 ms value held for 40 ms
after), which makes braindecode's own pad a no-op.  ``pad_mode`` and ``pad_align``
expose the alternatives and the realised fraction is in :meth:`describe`.

Why ``EEGPT.from_pretrained`` fails, and what actually works
------------------------------------------------------------
Reproduced on braindecode 1.7.0::

    EEGPT.from_pretrained("braindecode/eegpt-pretrained")
      -> RuntimeError: size mismatch for chans_id:
         checkpoint torch.Size([1, 62]) vs current model torch.Size([1, 19])
    EEGPT.from_pretrained(..., n_chans=64, n_times=120)
      -> ValueError: n_chans=64 different from chs_info=[... 62 entries ...]

Both come from the same root cause: ``chan_proj_type`` defaults to
``"conv1d_constraint"``, which routes the encoder through the 19-channel probe
montage and registers a 19-entry ``chans_id`` buffer, while the hub config pins
``chs_info`` to the checkpoint's 62 channels so ``n_chans`` may not be overridden.
The fix is to pass ``chan_proj_type="none"`` *and* let the wrapper own the channel
mapping.  ``InterpolatedEEGPT.from_pretrained(repo, chs_info=<our 64>, n_times=...,
chan_proj_type="none")`` does succeed -- but it loads with ``strict=False`` and would
not have complained had nothing loaded, so this module builds the backbone itself and
copies the checkpoint tensor by tensor with an asserted coverage report.  A randomly
initialised "foundation model" is worse than no baseline.

Amplitude convention -- the least-constrained choice on this rung
-----------------------------------------------------------------
The pipeline feeds robust-scaled units (sub-01 ``w0600``: mean -0.022, std 0.876,
p99 |x| 3.38), which are dimensionless.  EEGPT pretrained on data in physical units;
*which* physical unit is **inferred, not documented** -- neither the braindecode port
nor the HF model card states it, and this model family splits between millivolts
(most) and 0.1 mV (LaBraM).  ``input_scale`` bridges the two, and its default,
:data:`ROBUST_UNITS_TO_MILLIVOLTS`, is fixed by two *independent* measurements that
agree on millivolts:

* physical -- ``derived/scaling/sub-*_robust.npz`` gives a median robust sigma of
  **17.3 uV** across 80 subjects x 64 channels (p10 11.0, p90 28.9), so one unit is
  0.0173 mV;
* internal -- at ``input_scale=0.0173`` the patch-embedding token RMS (0.023) matches
  the *pretrained* channel-embedding RMS (0.0225), and the patch-embedding bias, to
  within 5 %.  At ``input_scale=1.0`` the data token is **20x** the channel embedding,
  so the per-electrode identity that EEGPT's paper credits with carrying its spatial
  information contributes ~0.2 % of the token variance.

This is still the weakest link: on frozen features, ``input_scale`` 1.0 vs 0.1 gives
cosine 0.43, whereas two *different trials* give 0.733 -- the knob moves the
representation more than the data does.  It is a-priori unit matching, not a tuned
value, and it belongs in the run report either way.

Fine-tuning knobs: ``freeze_backbone`` (EEGPT's own downstream protocol is a *frozen*
encoder + linear probe, so this is the faithful ablation), ``n_unfrozen_blocks`` for a
partial unfreeze of the top of the 8-block stack, and ``backbone_lr_scale``, which
:meth:`param_groups` turns into a real per-group learning rate.  ``trainer.py`` calls
``encoder.param_groups(lr, wd)`` when it exists, so this one is live -- verified in a
run log: ``encoder supplied 4 custom parameter group(s); LRs: [1e-05, 1e-05, 0.0001,
0.0001]`` at ``train.optimizer.lr = 1e-4``.
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

__all__ = [
    "EEGPTEncoder",
    "BIOSEMI64_CHANNEL_ORDER",
    "EEGPT_HF_REPO",
    "EEGPT_PRETRAIN_SFREQ",
    "ROBUST_UNIT_MICROVOLTS",
    "ROBUST_UNITS_TO_MILLIVOLTS",
    "load_pretrained_eegpt",
    "channel_mapping_report",
]

log = logging.getLogger(__name__)


#: Hugging Face repo carrying the EEGPT weights (25.3 M parameters).
EEGPT_HF_REPO = "braindecode/eegpt-pretrained"

#: Sampling rate the checkpoint was pretrained at (from its ``config.json``).
EEGPT_PRETRAIN_SFREQ = 250.0

#: Median robust sigma of our epochs in microvolts, over 80 subjects x 64 channels
#: (``derived/scaling/sub-*_robust.npz``; p10 11.0, p90 28.9).  One robust unit of the
#: data the trainer hands us is this many microvolts.
ROBUST_UNIT_MICROVOLTS = 17.27

#: Default :attr:`EEGPTEncoder.input_scale`: converts our robust-scaled units to
#: millivolts, the convention this model family pretrains in.  See the module
#: docstring -- it is also, independently, the scale at which the data token and
#: EEGPT's pretrained channel embedding have the same magnitude.
ROBUST_UNITS_TO_MILLIVOLTS = ROBUST_UNIT_MICROVOLTS / 1000.0

#: Canonical BioSemi-64 order (contract B, ``derived/channels.json``).  Kept here so the
#: encoder can be built without the data root mounted; :func:`_default_ch_names`
#: cross-checks it against ``channels.json`` whenever that file is reachable and raises
#: on disagreement -- a silently permuted montage is a silently wrong topography, and
#: here it would also silently mis-address EEGPT's per-electrode embeddings.
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

_CHANNEL_MODES = ("interpolate", "subset", "conv19")
_POOL_MODES = ("flatten", "mean_patch", "mean")
_PAD_MODES = ("replicate", "zeros", "reflect")
_PAD_ALIGNS = ("left", "center", "right")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _import_eegpt():
    """Import braindecode's EEGPT lazily, with an actionable error message."""
    try:
        from braindecode.models import EEGPT, InterpolatedEEGPT  # noqa: PLC0415
        from braindecode.models.eegpt import (  # noqa: PLC0415
            EEGPT_CHANNELS,
            _EEGPT_TARGET_CHS_INFO,
        )
    except Exception as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "the 'eegpt' encoder needs braindecode>=1.7 with Hugging Face Hub support "
            "(`pip install 'braindecode[hug]'`); do NOT let pip move torch/numpy/mne/"
            f"torchaudio while doing it. Original error: {exc}"
        ) from exc
    return EEGPT, InterpolatedEEGPT, list(EEGPT_CHANNELS), list(_EEGPT_TARGET_CHS_INFO)


def _default_ch_names(n_channels: int) -> List[str]:
    """The 64 BioSemi channel names in acquisition order, cross-checked against disk."""
    if n_channels != len(BIOSEMI64_CHANNEL_ORDER):
        raise ValueError(
            f"eegpt needs explicit ch_names for n_channels={n_channels}; only the "
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
                    f"{path} disagrees with BIOSEMI64_CHANNEL_ORDER in fm_eegpt.py. "
                    "The channel order decides which EEGPT channel embedding each "
                    "electrode reads, so this must not be papered over -- pass "
                    "ch_names=<the on-disk order> explicitly or fix the constant."
                )
            return [str(c) for c in on_disk]
    return names


def _chs_info(names: Sequence[str], montage: str = "biosemi64") -> List[Dict[str, Any]]:
    """MNE-style ``chs_info`` (``ch_name`` / ``kind`` / ``loc``) for ``names``.

    The 3-D positions are only used to build the spherical-spline interpolation matrix
    for the EEGPT channels our montage lacks (``PO5``/``PO6``); every name-matched
    channel is a one-hot copy and its position is ignored.
    """
    import mne  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    pos = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    lookup = {k.upper(): v for k, v in pos.items()}
    out: List[Dict[str, Any]] = []
    for name in names:
        loc = lookup.get(str(name).upper())
        if loc is None:
            raise ValueError(
                f"channel {name!r} is not in the {montage!r} montage, so no 3-D position "
                "is available for the EEGPT channel interpolation; pass channel_mode="
                "'subset' or a montage that contains it"
            )
        out.append({"ch_name": str(name), "kind": "eeg", "loc": np.asarray(loc, dtype=float)})
    return out


def channel_mapping_report(ch_names: Sequence[str]) -> Dict[str, Any]:
    """Which of ``ch_names`` EEGPT knows, and which of EEGPT's channels we lack.

    Pure bookkeeping, no model needed -- call it from a notebook to reproduce the
    numbers quoted in the module docstring.
    """
    _, _, eegpt_channels, _ = _import_eegpt()
    ours = [str(c) for c in ch_names]
    theirs_up = {c.upper() for c in eegpt_channels}
    ours_up = {c.upper() for c in ours}
    matched = [c for c in ours if c.upper() in theirs_up]
    return {
        "n_ours": len(ours),
        "n_eegpt": len(eegpt_channels),
        "matched": matched,
        "n_matched": len(matched),
        "ours_unknown_to_eegpt": [c for c in ours if c.upper() not in theirs_up],
        "eegpt_absent_from_ours": [c for c in eegpt_channels if c.upper() not in ours_up],
    }


def _fetch_checkpoint(repo_id: str, local_files_only: Optional[bool]) -> Dict[str, torch.Tensor]:
    """Download and read the checkpoint's raw ``state_dict`` (safetensors, then .bin)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    def _dl(filename: str, offline: Optional[bool]) -> str:
        kw: Dict[str, Any] = {} if offline is None else {"local_files_only": bool(offline)}
        return hf_hub_download(repo_id=repo_id, filename=filename, **kw)

    attempts = [local_files_only] if local_files_only is not None else [None, True]
    errors: List[str] = []
    for offline in attempts:
        for filename in ("model.safetensors", "pytorch_model.bin"):
            try:
                path = _dl(filename, offline)
            except Exception as exc:  # not present / no network
                errors.append(f"{filename}(local_files_only={offline}): {exc}")
                continue
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file  # noqa: PLC0415

                return dict(load_file(path))
            return dict(torch.load(path, map_location="cpu", weights_only=True))
    raise RuntimeError(
        f"could not fetch the EEGPT checkpoint from {repo_id!r}. Populate the cache from "
        f"a login node with HF_HOME={os.environ.get('HF_HOME')} set, or pass "
        f"local_files_only=False. Attempts: {errors}"
    )


def load_pretrained_eegpt(
    backbone: nn.Module,
    repo_id: str = EEGPT_HF_REPO,
    *,
    local_files_only: Optional[bool] = None,
    min_fraction: float = 0.999,
) -> Dict[str, Any]:
    """Transplant ``repo_id``'s weights into ``backbone`` and *prove* it happened.

    We do not use ``EEGPT.from_pretrained``: it raises on the default channel
    configuration (see the module docstring) and, when it does succeed, it loads with
    ``strict=False`` -- so a geometry change that silently skipped every tensor would
    look exactly like success.  Instead the backbone is built here at our geometry and
    the checkpoint is copied key by key with shape checks, then verified against the
    live module.

    ``chans_id`` is an *index* buffer (which row of ``chan_embed`` each input channel
    reads), not a learned tensor.  It is copied when the shapes agree and reported
    separately otherwise; it is excluded from the parameter fraction because it holds
    no information the channel names do not.

    Raises if fewer than ``min_fraction`` of the backbone's ``nn.Parameter`` elements
    came from the checkpoint.
    """
    src_sd = _fetch_checkpoint(repo_id, local_files_only)
    tgt_sd = backbone.state_dict()
    param_names = {name for name, _ in backbone.named_parameters()}

    staged: Dict[str, torch.Tensor] = {}
    skipped: List[Tuple[str, str]] = []
    for key, want in tgt_sd.items():
        have = src_sd.get(key)
        if have is None:
            skipped.append((key, "absent from checkpoint"))
        elif have.shape == want.shape:
            staged[key] = have.detach().clone().to(want.dtype)
        else:
            skipped.append((key, f"checkpoint {tuple(have.shape)} vs model {tuple(want.shape)}"))

    missing, unexpected = backbone.load_state_dict(staged, strict=False)
    live = backbone.state_dict()
    not_applied = [k for k in staged if not torch.equal(live[k].cpu(), staged[k].cpu())]
    if not_applied:  # pragma: no cover - would mean torch lied to us
        raise RuntimeError(f"load_state_dict silently dropped {len(not_applied)} tensors")

    n_total = sum(t.numel() for name, t in tgt_sd.items() if name in param_names)
    n_loaded = sum(tgt_sd[k].numel() for k in staged if k in param_names)
    frac = n_loaded / max(1, n_total)
    report = {
        "repo_id": repo_id,
        "checkpoint_tensors": len(src_sd),
        "tensors_total": len(tgt_sd),
        "tensors_from_checkpoint": len(staged),
        "tensors_skipped": skipped,
        "params_total": n_total,
        "params_from_checkpoint": n_loaded,
        "fraction": round(frac, 6),
        "chans_id_from_checkpoint": "chans_id" in staged,
        "load_state_dict_missing": list(missing),
        "load_state_dict_unexpected": list(unexpected),
    }
    if frac < float(min_fraction):
        raise RuntimeError(
            f"only {frac:.4%} of the EEGPT backbone came from {repo_id!r} "
            f"({n_loaded:,}/{n_total:,} params; skipped={skipped[:6]}). This is a "
            "randomly initialised transformer, not a foundation model. Fix the "
            "geometry/channel mode, or pass pretrained=False deliberately."
        )
    return report


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------


@register_eeg_encoder("eegpt", "fm_eegpt", "eegpt_ft")
class EEGPTEncoder(EEGEncoder):
    """Fine-tuned EEGPT trunk + the standard TACTUS projection head (contract F).

    Parameters
    ----------
    pretrained:
        Load ``hf_repo`` into the backbone.  ``False`` is only legal together with
        ``allow_random_init=True`` -- see :func:`load_pretrained_eegpt`.
    hf_repo, local_files_only:
        Checkpoint location; ``local_files_only=True`` forces the ``HF_HOME`` cache
        (use it on compute nodes without outbound network).
    channel_mode:
        ``"interpolate"`` (default, ``InterpolatedEEGPT`` + ``name_match``: 60 one-hot
        copies, ``PO5``/``PO6`` spline-interpolated), ``"subset"`` (feed only the 60
        name-matched electrodes) or ``"conv19"`` (braindecode's default learned
        64 -> 19 projection; not recommended, see the module docstring).
    interpolation_mode, interpolation_method, trainable_interpolation:
        Forwarded to :class:`~braindecode.modules.ChannelInterpolationLayer`.
        ``interpolation_mode="always"`` re-interpolates *every* channel including the
        60 EEGPT already knows -- avoid unless you are studying the mapping itself.
    ch_names, montage:
        Channel names in the order of the data's channel axis (defaults to the
        BioSemi-64 order cross-checked against ``derived/channels.json``) and the MNE
        montage supplying their 3-D positions.
    resample_sfreq:
        ``None`` (default) feeds the native ``data_sfreq``.  ``250`` linearly upsamples
        each epoch inside the encoder so ``patch_size=64`` spans the same 256 ms it did
        during pretraining.  The *window* is identical either way.
    data_sfreq:
        Sampling rate of the incoming epochs (200 Hz, contract B).
    patch_size, patch_stride:
        ``None`` takes the checkpoint's geometry (64 / 32).  ``patch_size`` may not be
        changed while ``pretrained=True``: the patch embedding is a
        ``Conv2d(1, 512, (1, 64))`` and a different kernel width discards it.
    pad_mode, pad_align:
        How an epoch is padded onto the patch grid.  ``"replicate"``/``"left"`` by
        default; braindecode's own fallback is ``zeros``/right, which fabricates a step
        on data that carry no baseline correction.
    pool:
        ``"flatten"`` (default, ``n_patches * embed_num * embed_dim``; the view EEGPT's
        own linear probe uses and the only one that preserves patch order -- note the
        transformer attends *within* a patch across channels, never across patches, so
        all temporal structure lives in this axis), ``"mean_patch"`` (mean over
        patches, ``embed_num * embed_dim = 2048``) or ``"mean"`` (mean over patches and
        summary tokens, ``embed_dim = 512``).
    input_scale:
        Multiplies the input.  Default :data:`ROBUST_UNITS_TO_MILLIVOLTS` (0.01727)
        converts our robust-scaled units to millivolts; ``1.0`` feeds them raw and puts
        the data token 20x above EEGPT's channel embedding.  See the module docstring --
        this is the least-constrained parameter on the rung and must be reported.
    freeze_backbone, n_unfrozen_blocks:
        ``freeze_backbone=True`` reproduces EEGPT's *own* downstream protocol (frozen
        encoder + trained probe).  ``n_unfrozen_blocks=k`` freezes the trunk then
        re-enables the top ``k`` of 8 transformer blocks plus the final norm.
    backbone_lr_scale:
        Multiplier for the backbone's learning rate, consumed by :meth:`param_groups`.
    drop_prob, attn_drop_rate, drop_path_rate:
        Regularisation.  Defaults are EEGPT's own (all 0.0); none change a shape.
    allow_random_init:
        Escape hatch for an ablation that deliberately wants the architecture without
        the weights.  Emits a loud warning; never use it for the reported rung.
    proj_hidden, proj_blocks, proj_dropout:
        Residual projector geometry, as in :class:`~tactus.models.eeg.tsconv.TSConvEncoder`.
    **base:
        :class:`~tactus.models.eeg.base.EEGEncoder` arguments -- ``n_channels``,
        ``n_times``, ``embed_dim``, ``n_subjects``, ``subject_cond``, ...
    """

    #: config spellings used by configs/*.yaml
    config_aliases = {
        "hf_repo": ("repo_id", "checkpoint", "weights"),
        "proj_hidden": ("d_model", "d_hidden"),
        "proj_dropout": ("dropout", "drop"),
        "n_unfrozen_blocks": ("unfreeze_last_n", "n_trainable_blocks"),
        "backbone_lr_scale": ("backbone_lr_mult", "lr_scale_backbone"),
        "resample_sfreq": ("target_sfreq", "model_sfreq"),
        "data_sfreq": ("sfreq",),
        "channel_mode": ("channel_mapping", "chan_mode"),
    }

    def __init__(
        self,
        *,
        pretrained: bool = True,
        hf_repo: str = EEGPT_HF_REPO,
        local_files_only: Optional[bool] = None,
        channel_mode: str = "interpolate",
        interpolation_mode: str = "name_match",
        interpolation_method: str = "spline",
        trainable_interpolation: bool = False,
        ch_names: Optional[Sequence[str]] = None,
        montage: str = "biosemi64",
        resample_sfreq: Optional[float] = None,
        data_sfreq: float = 200.0,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        pad_mode: str = "replicate",
        pad_align: str = "left",
        pool: str = "flatten",
        input_scale: float = ROBUST_UNITS_TO_MILLIVOLTS,
        freeze_backbone: bool = False,
        n_unfrozen_blocks: Optional[int] = None,
        backbone_lr_scale: float = 0.1,
        drop_prob: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        allow_random_init: bool = False,
        min_pretrained_fraction: float = 0.999,
        proj_hidden: Optional[int] = None,
        proj_blocks: int = 1,
        proj_dropout: float = 0.5,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        EEGPT, InterpolatedEEGPT, eegpt_channels, target_chs = _import_eegpt()

        if channel_mode not in _CHANNEL_MODES:
            raise ValueError(f"channel_mode must be one of {_CHANNEL_MODES}, got {channel_mode!r}")
        if pool not in _POOL_MODES:
            raise ValueError(f"pool must be one of {_POOL_MODES}, got {pool!r}")
        if pad_mode not in _PAD_MODES:
            raise ValueError(f"pad_mode must be one of {_PAD_MODES}, got {pad_mode!r}")
        if pad_align not in _PAD_ALIGNS:
            raise ValueError(f"pad_align must be one of {_PAD_ALIGNS}, got {pad_align!r}")
        self.channel_mode = str(channel_mode)
        self.pool = str(pool)
        self.pad_mode = str(pad_mode)
        self.pad_align = str(pad_align)
        self.input_scale = float(input_scale)
        self.backbone_lr_scale = float(backbone_lr_scale)
        self.hf_repo = str(hf_repo)
        self.data_sfreq = float(data_sfreq)

        # ---- checkpoint geometry (64 / 32; hard-coded in the checkpoint's conv kernel) --
        ckpt_patch, ckpt_stride = 64, 32
        if patch_size is None:
            patch_size = ckpt_patch
        elif pretrained and int(patch_size) != ckpt_patch:
            raise ValueError(
                f"patch_size={patch_size} != the checkpoint's {ckpt_patch}. EEGPT's patch "
                f"embedding is a Conv2d(1, 512, (1, {ckpt_patch})); a different kernel width "
                "cannot load it. Resample or pad the epoch instead."
            )
        self.patch_size = int(patch_size)
        self.patch_stride = int(ckpt_stride if patch_stride is None else patch_stride)
        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {self.patch_stride}")

        # ---- time axis: optional resample, then pad onto the patch grid -----------------
        self.resample_sfreq = None if resample_sfreq is None else float(resample_sfreq)
        if self.resample_sfreq is not None and self.resample_sfreq <= 0:
            raise ValueError(f"resample_sfreq must be positive, got {resample_sfreq}")
        self.model_sfreq = float(self.resample_sfreq or self.data_sfreq)
        self.resampled_n_times = self._resampled_len(self.n_times)
        self.n_patches, self.backbone_n_times = self._patch_grid(self.resampled_n_times)
        self.pad_total = self.backbone_n_times - self.resampled_n_times
        self.pad_fraction = self.pad_total / self.backbone_n_times

        # ---- channels -------------------------------------------------------------------
        names = [str(c) for c in ch_names] if ch_names is not None else _default_ch_names(self.n_channels)
        if len(names) != self.n_channels:
            raise ValueError(f"ch_names has {len(names)} entries but n_channels={self.n_channels}")
        self.ch_names: List[str] = names
        self.channel_report = channel_mapping_report(names)
        known_up = {c.upper() for c in eegpt_channels}

        common: Dict[str, Any] = dict(
            n_outputs=1,                 # unused: return_encoder_output makes the head Identity
            n_times=self.backbone_n_times,
            sfreq=self.model_sfreq,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            drop_prob=float(drop_prob),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=float(drop_path_rate),
            return_encoder_output=True,
        )
        self.register_buffer("_subset_index", torch.zeros(0, dtype=torch.long), persistent=False)
        if channel_mode == "interpolate":
            self.backbone = InterpolatedEEGPT(
                chs_info=_chs_info(names, montage),
                chan_proj_type="none",
                interpolation_mode=str(interpolation_mode),
                interpolation_method=str(interpolation_method),
                trainable=bool(trainable_interpolation),
                **common,
            )
            self.backbone_channels = [str(c["ch_name"]) for c in target_chs]
        elif channel_mode == "subset":
            keep = [i for i, c in enumerate(names) if c.upper() in known_up]
            if not keep:
                raise ValueError("channel_mode='subset' matched none of EEGPT's channels")
            self.register_buffer("_subset_index", torch.tensor(keep, dtype=torch.long), persistent=False)
            kept_names = [names[i] for i in keep]
            self.backbone = EEGPT(
                chs_info=_chs_info(kept_names, montage),
                chan_proj_type="none",
                **common,
            )
            self.backbone_channels = kept_names
        else:  # conv19
            self.backbone = EEGPT(
                n_chans=self.n_channels,
                chan_proj_type="conv1d_constraint",
                n_chans_target=19,
                **common,
            )
            self.backbone_channels = list(getattr(self.backbone, "channel_names", []))

        if tuple(self.backbone.target_encoder.num_patches)[1] != self.n_patches:
            raise RuntimeError(  # pragma: no cover - arithmetic guard
                f"patch-grid arithmetic disagrees with braindecode: we computed "
                f"{self.n_patches} patches, it built "
                f"{tuple(self.backbone.target_encoder.num_patches)[1]}"
            )
        if int(self.backbone.target_encoder.patch_embed.padding_size) != 0:
            raise RuntimeError(  # pragma: no cover - arithmetic guard
                "braindecode still wants to zero-pad "
                f"{self.backbone.target_encoder.patch_embed.padding_size} samples; our "
                "pre-padding was supposed to make that a no-op"
            )

        # ---- weights ---------------------------------------------------------------------
        if pretrained:
            self.pretrained_report = load_pretrained_eegpt(
                self.backbone,
                hf_repo,
                local_files_only=local_files_only,
                min_fraction=float(min_pretrained_fraction),
            )
        elif allow_random_init:
            warnings.warn(
                "EEGPTEncoder built with pretrained=False: this is a randomly initialised "
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

        # ---- freezing ---------------------------------------------------------------------
        self.freeze_backbone = bool(freeze_backbone)
        self.n_unfrozen_blocks = None if n_unfrozen_blocks is None else int(n_unfrozen_blocks)
        self._apply_freezing()

        self.proj_hidden = proj_hidden
        self.proj_blocks = int(proj_blocks)
        self.proj_dropout = float(proj_dropout)
        self.setup_head()

        # The trainer does not call describe(), so the interface decision is logged here:
        # the pretrained fraction, the padding fraction and the channel cost must appear
        # in every run's log.
        rep = self.channel_report
        log.info(
            "EEGPT: backbone %.2fM params, %.3f%% from %s (%s tensors) | epoch %d @ %.0f Hz "
            "-> %d @ %.0f Hz -> %d patches x %d (stride %d) = %d samples, %d padded "
            "(%.2f%% %s/%s) | patch = %.0f ms (pretrained 256 ms @ 250 Hz) | channels %s: "
            "%d/%d matched, dropped %s, interpolated %s | input_scale=%.5g (%.3g uV per "
            "unit) | pool=%s dim=%d | trainable backbone %.2fM | INDICATIVE baseline, "
            "3 folds",
            count_parameters(self.backbone, trainable_only=False) / 1e6,
            100.0 * float(self.pretrained_report.get("fraction", 0.0)),
            self.pretrained_report.get("repo_id"),
            f"{self.pretrained_report.get('tensors_from_checkpoint')}/"
            f"{self.pretrained_report.get('tensors_total')}",
            self.n_times, self.data_sfreq, self.resampled_n_times, self.model_sfreq,
            self.n_patches, self.patch_size, self.patch_stride, self.backbone_n_times,
            self.pad_total, 100.0 * self.pad_fraction, self.pad_mode, self.pad_align,
            1000.0 * self.patch_size / self.model_sfreq,
            self.channel_mode, rep["n_matched"], rep["n_ours"],
            rep["ours_unknown_to_eegpt"],
            rep["eegpt_absent_from_ours"] if self.channel_mode == "interpolate" else [],
            self.input_scale, ROBUST_UNIT_MICROVOLTS,
            self.pool, self.feature_dim,
            count_parameters(self.backbone, trainable_only=True) / 1e6,
        )

    # ---------------------------------------------------------------------------------
    # construction helpers
    # ---------------------------------------------------------------------------------

    def _resampled_len(self, n_times: int) -> int:
        """Length of ``n_times`` samples after the (optional) resample to ``model_sfreq``."""
        if self.resample_sfreq is None:
            return int(n_times)
        return max(1, int(round(int(n_times) * self.model_sfreq / self.data_sfreq)))

    def _patch_grid(self, n_times: int) -> Tuple[int, int]:
        """``(n_patches, padded_length)`` for the smallest grid covering ``n_times``."""
        n_times = int(n_times)
        if n_times <= self.patch_size:
            return 1, self.patch_size
        n_patches = math.ceil((n_times - self.patch_size) / self.patch_stride) + 1
        return int(n_patches), int(self.patch_size + (n_patches - 1) * self.patch_stride)

    def _apply_freezing(self) -> None:
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if self.n_unfrozen_blocks is not None:
            k = self.n_unfrozen_blocks
            blocks = list(self.backbone.target_encoder.blocks)
            if not 0 <= k <= len(blocks):
                raise ValueError(f"n_unfrozen_blocks must be in [0, {len(blocks)}], got {k}")
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for blk in blocks[len(blocks) - k:] if k else []:
                for p in blk.parameters():
                    p.requires_grad_(True)
            norm = getattr(self.backbone.target_encoder, "norm", None)
            if isinstance(norm, nn.Module):
                for p in norm.parameters():
                    p.requires_grad_(True)

    # ---------------------------------------------------------------------------------
    # trunk
    # ---------------------------------------------------------------------------------

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        """Linearly resample ``(B, C, T)`` from ``data_sfreq`` to ``model_sfreq``.

        Only ever an *up*-sample in the configured setting (200 -> 250 Hz), so no
        anti-alias filter is needed and no information is lost; the window in
        milliseconds is unchanged.
        """
        if self.resample_sfreq is None:
            return x
        target = self._resampled_len(x.shape[-1])
        if target == x.shape[-1]:
            return x
        return F.interpolate(x, size=target, mode="linear", align_corners=True)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad ``(B, C, T)`` onto :attr:`backbone_n_times`, the fixed patch grid."""
        t = int(x.shape[-1])
        target = self.backbone_n_times
        if t == target:
            return x
        if t > target:
            raise ValueError(
                f"EEGPTEncoder was built for at most {target} samples "
                f"({self.n_patches} patches of {self.patch_size} at stride "
                f"{self.patch_stride}) but got {t}. Rebuild the encoder with a larger "
                "n_times."
            )
        total = target - t
        if self.pad_align == "center":
            left = total // 2
        elif self.pad_align == "left":  # data first, padding after the window
            left = 0
        else:  # "right": padding first, data last
            left = total
        right = total - left
        mode = self.pad_mode
        if mode == "reflect" and max(left, right) >= t:
            mode = "replicate"  # torch requires pad < input length for reflect
        if mode == "zeros":
            return F.pad(x, (left, right), mode="constant", value=0.0)
        return F.pad(x, (left, right), mode=mode)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self.input_scale != 1.0:
            x = x * self.input_scale
        if self.channel_mode == "subset":
            x = x.index_select(1, self._subset_index)
        x = self._pad(self._resample(x))
        # (B, n_patches, embed_num * embed_dim); EEGPT attends across channels *within*
        # a patch only, so the patch axis is the whole of its temporal structure.
        h = self.backbone(x, return_features=True)["features"]
        if self.pool == "flatten":
            return h.flatten(1)
        if self.pool == "mean_patch":
            return h.mean(dim=1)
        return h.reshape(h.shape[0], h.shape[1], -1, self.backbone.embed_dim).mean(dim=(1, 2))

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

        ``tactus/train/trainer.py`` calls this (``pg_fn(lr, wd)``) whenever the encoder
        defines it, and when it does the trainer takes the *encoder's* parameters only
        from these groups -- so every trainable encoder parameter must appear here
        exactly once or it silently stops being optimised.  The buckets below iterate
        ``self.named_parameters()``, which covers trunk, conditioner and projection head.

        A gradient hook would NOT be a substitute for the per-group ``lr``: AdamW
        normalises by the gradient's own second moment, so scaling gradients does not
        scale the step.

        Verified in a real run log at ``train.optimizer.lr = 1e-4``::

            encoder supplied 4 custom parameter group(s); LRs: [1e-05, 1e-05, 0.0001, 0.0001]
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
        chan = dict(self.channel_report)
        dropped = list(chan["ours_unknown_to_eegpt"])
        interpolated = (
            list(chan["eegpt_absent_from_ours"]) if self.channel_mode == "interpolate" else []
        )
        out.update(
            {
                "backbone": "braindecode.models.EEGPT (EEGPT, Wang et al. NeurIPS 2024)",
                "pretrained_repo": rep.get("repo_id"),
                "pretrained_param_fraction": rep.get("fraction"),
                "pretrained_tensors": (
                    f"{rep.get('tensors_from_checkpoint')}/{rep.get('tensors_total')}"
                ),
                "pretrained_tensors_skipped": rep.get("tensors_skipped"),
                "backbone_parameters": count_parameters(self.backbone, trainable_only=False),
                "backbone_trainable": count_parameters(self.backbone, trainable_only=True),
                "patch_size": self.patch_size,
                "patch_stride": self.patch_stride,
                "n_patches": self.n_patches,
                "epoch_samples": self.n_times,
                "data_sfreq": self.data_sfreq,
                "model_sfreq": self.model_sfreq,
                "resample_sfreq": self.resample_sfreq,
                "resampled_samples": self.resampled_n_times,
                "backbone_samples": self.backbone_n_times,
                "pad_samples": self.pad_total,
                "pad_fraction": round(self.pad_fraction, 4),
                "pad_mode": self.pad_mode,
                "pad_align": self.pad_align,
                "patch_ms": round(1000.0 * self.patch_size / self.model_sfreq, 1),
                "pretrain_patch_ms": round(1000.0 * self.patch_size / EEGPT_PRETRAIN_SFREQ, 1),
                "channel_mode": self.channel_mode,
                "channels_in": chan["n_ours"],
                "channels_to_backbone": len(self.backbone_channels),
                "channels_matched": chan["n_matched"],
                "channels_dropped": dropped,
                "channels_interpolated": interpolated,
                "pool": self.pool,
                "input_scale": self.input_scale,
                "input_units": (
                    f"backbone input = {self.input_scale:.5g} x robust sigma; 1 robust "
                    f"sigma ~= {ROBUST_UNIT_MICROVOLTS:.1f} uV (median over 80 subjects x "
                    f"64 channels), so 1.0 at the backbone's input is "
                    f"~{ROBUST_UNIT_MICROVOLTS / max(self.input_scale, 1e-12):.4g} uV "
                    f"(= millivolts when input_scale is {ROBUST_UNITS_TO_MILLIVOLTS:.5g})"
                ),
                "freeze_backbone": self.freeze_backbone,
                "n_unfrozen_blocks": self.n_unfrozen_blocks,
                "backbone_lr_scale": self.backbone_lr_scale,
                "interface_note": (
                    f"EEGPT patch = {self.patch_size} samples = "
                    f"{1000.0 * self.patch_size / self.model_sfreq:.0f} ms at "
                    f"{self.model_sfreq:.0f} Hz (256 ms at the 250 Hz it was pretrained "
                    f"at{'' if self.resample_sfreq else '; NOT corrected -- the model sees our signal 1.25x slower than its pretraining data'}). "
                    f"The {self.n_times}-sample epoch needs {self.pad_total} samples of "
                    f"{self.pad_mode} padding ({self.pad_fraction:.2%}) to fill "
                    f"{self.n_patches} patches. Channel mapping '{self.channel_mode}': "
                    f"{chan['n_matched']}/{chan['n_ours']} electrodes passed through by "
                    f"name, {dropped} effectively dropped (EEGPT has no embedding for "
                    f"them), {interpolated} spline-interpolated from our montage."
                ),
            }
        )
        return out


if __name__ == "__main__":  # pragma: no cover - CPU smoke check for the server-side run
    import argparse

    ap = argparse.ArgumentParser(description="EEGPT encoder smoke test (CPU)")
    ap.add_argument("--n-times", type=int, default=120)
    ap.add_argument("--pool", default="flatten")
    ap.add_argument("--channel-mode", default="interpolate", choices=list(_CHANNEL_MODES))
    ap.add_argument("--resample-sfreq", type=float, default=None)
    ap.add_argument("--conds", default="none,subject_token,subject_layer,sulora")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for cond in args.conds.split(","):
        enc = EEGPTEncoder(
            n_times=args.n_times,
            embed_dim=256,
            pool=args.pool,
            channel_mode=args.channel_mode,
            resample_sfreq=args.resample_sfreq,
            subject_cond=cond,
        ).eval()
        x = torch.randn(4, 64, args.n_times)
        sid = torch.tensor([1, 2, 3, -1])
        with torch.no_grad():
            z = enc(x, sid)
        d = enc.describe()
        print(
            f"{cond:>14s}  params={count_parameters(enc):>11,d}  out={tuple(z.shape)}  "
            f"norm={z.norm(dim=-1).mean():.4f}  pretrained={d['pretrained_param_fraction']}  "
            f"pad={d['pad_fraction']}  dropped={d['channels_dropped']}  "
            f"rule={enc.unseen_subject_state()['rule']}"
        )
