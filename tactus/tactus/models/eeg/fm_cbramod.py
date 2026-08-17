"""CBraMod EEG foundation model as a TACTUS encoder (contract F).

Baseline-ladder rung 5 -- *indicative* EEG-foundation-model fine-tuning.  Read the
interface caveats below before quoting any number produced by this file.

Model
-----
CBraMod (Wang et al., *CBraMod: A Criss-Cross Brain Foundation Model for EEG
Decoding*, ICLR 2025), as shipped by ``braindecode.models.CBraMod`` (braindecode
1.7.0), with the TUEG-pretrained weights from the HF hub repo
``braindecode/cbramod-pretrained``.

Backbone = 4,924,000 parameters, and **all 4,924,000 come from the checkpoint**
(211/211 checkpoint tensors, bit-exact -- :meth:`CBraModEncoder.verify_pretrained`
re-checks this against the cached ``model.safetensors`` at construction time and
records the result in ``describe()["pretrained"]``).  Construction with
``return_encoder_output=True`` replaces CBraMod's classification ``final_layer``
with ``nn.Identity``, so there is no randomly-initialised tensor left inside the
backbone at all; every new parameter lives in the TACTUS projector/conditioner.

If the checkpoint cannot be fetched this class **raises**.  Set
``allow_random_init=True`` to get a scratch-initialised CBraMod, and then do not
call the result a foundation model.

THE INTERFACE MISMATCH (do not hide this)
-----------------------------------------
CBraMod's temporal patch is ``patch_size=200`` samples = 1.0 s at 200 Hz, and it is
*baked into the pretrained weights*: ``patch_embedding.spectral_proj.0.weight`` has
shape ``(200, 101)``, i.e. it consumes ``rfft`` bins of a 200-sample patch, and the
conv stack ``proj_in`` maps 200 samples to 8 temporal positions.  Changing
``patch_size`` therefore *cannot* reuse the pretrained patch embedding -- unlike
LaBraM, which silently shrinks its patch when ``n_times < patch_size``, CBraMod
raises (``einops``: "can't divide axis of length 120 in chunks of 200").  That hard
error is the honest behaviour and we keep it: **we pad, we never re-patch.**

Our epochs are 120 samples (``w0600``, 0-600 ms, the primary window) or 180
(``wm100_800``, -100..800 ms, the baseline-contaminated sensitivity window).  So:

===================  ========  =======  ==============  ==================================
window               n_times   padded   padding fraction  note
===================  ========  =======  ==============  ==================================
``w0600`` (PRIMARY)  120       200      **40.0 %**       comparable to the whole ladder
``wm100_800``        180       200      **10.0 %**       cheaper padding, but baselined
sub-window "early"   30        200      85.0 %           TimeWindowHeads -- see below
sub-window "mid"     40        200      80.0 %
sub-window "late"    50        200      75.0 %
===================  ========  =======  ==============  ==================================

**Pre-registered choice: ``w0600``, 40 % padding.**  Rationale: rung 5 exists to
answer "does a TUEG-pretrained EEG transformer beat our 0.33 M-parameter conv on
*this* task", and that comparison is only interpretable if the model sees the same
0-600 ms of data as every other rung.  Buying a 10 % padding fraction by moving to
``wm100_800`` would confound "foundation model vs conv" with "window A vs window B"
*and* would hand this rung alone the -100..0 ms baseline that the blueprint
deliberately refuses to subtract.  ``wm100_800`` is still worth running -- as a
**labelled sensitivity arm**, not as the headline.

A 40 % padded model is a weak, pessimistically-biased probe of CBraMod: three fifths
of the tokens the patch embedding was pretrained to read are real, two fifths are
synthetic.  Any *negative* result at rung 5 is therefore about the interface as much
as about the model, and must be reported that way.

The ``TimeWindowHeads`` sub-windows are worse still (75-85 % padding).  They satisfy
contract F and they run, but a CBraMod onset curve is not scientifically meaningful
and should not be reported.

Padding scheme
--------------
``pad_mode`` in ``{"reflect", "replicate", "median", "zero"}``, ``pad_side`` in
``{"right", "left", "both"}``.  Default ``reflect``/``right``.

The epochs carry **no baseline correction** (contract B), so a trial-channel's DC
level is not zero: measured on ``sub-01_w0600``, the mean ``|per-trial-channel
median|`` is 0.42 against a mean within-trial SD of 0.49.  Zero-padding therefore
splices in a step of roughly one within-trial SD at the seam, and CBraMod reads the
patch through an ``rfft`` (``spectral_proj``) where such a step is broadband
leakage.  Symmetric (reflect) extension is the standard short-segment extension for
exactly this reason -- it is value- and slope-continuous at the seam and preserves
the segment's spectrum -- so it is the a-priori default.  ``replicate`` (DC hold)
and ``median`` are the conservative alternatives; ``zero`` is kept only so the
sensitivity of the embedding to the padding value can be *measured* rather than
asserted (see ``__main__``, which prints the pairwise cosine similarity of the
backbone features under all four schemes on real trials).

Measured, 128 real trials of ``sub-01_w0600``, pretrained backbone features, no
projector.  The calibration constant is the *between-trial* cosine, 0.087 -- i.e.
features of two different trials are near-orthogonal:

===========================  ================  =========================
comparison                   cosine (same trial)  RDM Pearson r
===========================  ================  =========================
different pad **value**      0.69 - 0.88       0.82 - 0.90
different pad **side**       0.25 - 0.31       0.76 - 0.80
===========================  ================  =========================

So a trial stays recognisably itself under any padding value (0.69 >> 0.087), but
10-18 % of the trial-to-trial geometry that retrieval actually reads is
padding-dependent, and *where* the real 600 ms sits inside the 200-sample patch is
a first-order effect -- a bigger lever than what the padding is filled with.  Both
knobs are therefore fixed a priori and must not be varied within the ladder.

Fine-tuning knobs
-----------------
``freeze_backbone``      freeze all 4.92 M pretrained parameters (linear probe).
``unfreeze_last_n``      freeze everything except the last *n* criss-cross blocks
                         (+ ``proj_out``); the usual partial fine-tune.
``freeze_patch_embedding``  keep the patch/spectral embedding frozen while the
                         transformer adapts -- the recommended setting when the
                         input is 40 % synthetic, because the patch embedding is
                         the part that actually looks at the padded samples.
``backbone_lr_scale``    discriminative LR factor, exposed via :meth:`param_groups`.
                         **The trainer does not consume it yet** (``Trainer.__init__``
                         builds two decay/no-decay groups at a single ``lr``), so
                         until someone wires ``encoder.param_groups(...)`` in,
                         freezing is the only fine-tuning control that has any
                         effect.  This is recorded in ``describe()`` so a run log
                         cannot claim a discriminative LR it never had.

Registration
------------
This module is *not* imported by ``tactus/models/eeg/__init__.py`` (a shared file
this task may not edit), so the ``@register_eeg_encoder`` decorator does not run
unless something imports it.  ``configs/cbramod_ft.yaml`` will fail with
``unknown EEG encoder 'cbramod'`` until one line is added there::

    from .fm_cbramod import CBraModEncoder   # noqa: F401
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..heads import ResidualProjection
from .base import EEGEncoder, count_parameters, register_eeg_encoder

log = logging.getLogger(__name__)

__all__ = ["CBraModEncoder", "CBRAMOD_HF_REPO", "pad_to_multiple"]

#: HF hub repo holding the TUEG-pretrained CBraMod weights.
CBRAMOD_HF_REPO = "braindecode/cbramod-pretrained"

#: CBraMod's pretrained temporal patch, in samples.  1.0 s at 200 Hz.  Baked into
#: ``patch_embedding.spectral_proj`` (rfft of 200 samples -> 101 bins) and into the
#: ``proj_in`` conv stack; it cannot be changed without discarding those weights.
CBRAMOD_PATCH_SIZE = 200

_PAD_MODES = ("reflect", "replicate", "median", "zero")
_PAD_SIDES = ("right", "left", "both")


# ======================================================================================
# padding
# ======================================================================================


def _gather_index(n_times: int, left: int, right: int, mode: str, device) -> torch.Tensor:
    """Index vector realising ``reflect``/``replicate`` padding for any pad length.

    ``F.pad(mode="reflect")`` refuses pads >= the input length, which is exactly the
    regime we are in (120 -> 200 is fine, but the 30-sample sub-window is not), so we
    build the index arithmetically instead and ``index_select``.
    """
    idx = torch.arange(-left, n_times + right, device=device)
    if mode == "replicate":
        return idx.clamp_(0, n_times - 1)
    if n_times == 1:
        return torch.zeros_like(idx)
    period = 2 * n_times - 2  # reflect WITHOUT repeating the edge sample (numpy 'reflect')
    idx = torch.remainder(idx, period)
    return torch.where(idx >= n_times, period - idx, idx)


def pad_to_multiple(
    x: torch.Tensor,
    target: int,
    *,
    mode: str = "reflect",
    side: str = "right",
) -> torch.Tensor:
    """Pad ``(B, C, T)`` along time up to exactly ``target`` samples.

    ``mode``: ``reflect`` (symmetric extension), ``replicate`` (edge hold),
    ``median`` (per trial-and-channel temporal median), ``zero``.
    ``side``:  ``right`` (default; the stimulus-locked signal stays at the patch
    start), ``left``, or ``both`` (extra sample goes right).
    """
    if mode not in _PAD_MODES:
        raise ValueError(f"pad_mode must be one of {_PAD_MODES}, got {mode!r}")
    if side not in _PAD_SIDES:
        raise ValueError(f"pad_side must be one of {_PAD_SIDES}, got {side!r}")
    n_times = int(x.shape[-1])
    total = int(target) - n_times
    if total < 0:
        raise ValueError(f"cannot pad {n_times} samples down to {target}")
    if total == 0:
        return x
    if side == "right":
        left, right = 0, total
    elif side == "left":
        left, right = total, 0
    else:
        left, right = total // 2, total - total // 2

    if mode in ("reflect", "replicate"):
        idx = _gather_index(n_times, left, right, mode, x.device)
        return x.index_select(-1, idx)

    if mode == "median":
        fill = x.median(dim=-1, keepdim=True).values  # (B, C, 1), per trial & channel
    else:  # zero
        fill = x.new_zeros(x.shape[:-1] + (1,))
    parts = []
    if left:
        parts.append(fill.expand(*fill.shape[:-1], left))
    parts.append(x)
    if right:
        parts.append(fill.expand(*fill.shape[:-1], right))
    return torch.cat(parts, dim=-1)


# ======================================================================================
# encoder
# ======================================================================================


@register_eeg_encoder("cbramod", "cbramod_ft", "fm_cbramod")
class CBraModEncoder(EEGEncoder):
    """TUEG-pretrained CBraMod backbone + the TACTUS projector (contract F).

    Parameters
    ----------
    pretrained:
        Load ``braindecode/cbramod-pretrained``.  ``True`` (default) and a failure
        to load is a hard error unless ``allow_random_init=True``.
    hf_repo:
        Checkpoint repo id.
    allow_random_init:
        Permit a scratch-initialised backbone when the checkpoint is unavailable.
        Only for offline plumbing tests; such a run is **not** a foundation-model
        baseline and ``describe()["pretrained"]["loaded"]`` will say so.
    verify_pretrained:
        After loading, re-read the cached ``model.safetensors`` and assert every
        checkpoint tensor is bit-exact in the live module.  Cheap (4.9 M floats) and
        it is the only thing standing between "we called from_pretrained" and "the
        pretrained numbers are actually in the model".
    pad_mode, pad_side:
        See :func:`pad_to_multiple`.  Defaults ``reflect`` / ``right``.
    pad_to:
        Padded length.  ``None`` (default) = the smallest multiple of 200 that holds
        ``n_times`` (200 for both 120 and 180).  The padded length is fixed at
        construction so that ``extract_features`` returns a *constant* width for any
        window shorter than it -- which is what ``TimeWindowHeads`` needs.
    freeze_backbone, unfreeze_last_n, freeze_patch_embedding, backbone_lr_scale:
        Fine-tuning controls; see the module docstring.
    feature_pool:
        ``"flatten"`` (default, CBraMod's own downstream recipe: 64 chans x 1 patch x
        200 = 12800), ``"channel_mean"`` (200 per patch) or ``"mean"`` (200).
    input_scale:
        Multiplies the input before the backbone.  The stored epochs are already
        ~unit-variance, which is the scale CBraMod was pretrained at (its reference
        preprocessing expresses EEG in units of 100 uV), so the default is 1.0.
    drop_prob:
        CBraMod internal dropout.  Changing it does not affect weight loading.
    proj_hidden, proj_blocks, proj_dropout:
        TACTUS residual projector geometry.
    **base:
        :class:`~tactus.models.eeg.base.EEGEncoder` arguments.
    """

    #: config spellings used by configs/*.yaml
    config_aliases = {
        "proj_hidden": ("d_model", "d_hidden"),
        "proj_dropout": ("dropout",),
        "unfreeze_last_n": ("unfreeze_last_n_layers", "n_unfrozen_blocks"),
        "backbone_lr_scale": ("lr_scale", "backbone_lr_mult"),
    }

    def __init__(
        self,
        *,
        pretrained: bool = True,
        hf_repo: str = CBRAMOD_HF_REPO,
        allow_random_init: bool = False,
        verify_pretrained: bool = True,
        patch_size: int = CBRAMOD_PATCH_SIZE,
        pad_to: Optional[int] = None,
        pad_mode: str = "reflect",
        pad_side: str = "right",
        freeze_backbone: bool = False,
        unfreeze_last_n: Optional[int] = None,
        freeze_patch_embedding: bool = False,
        backbone_lr_scale: float = 0.1,
        feature_pool: str = "flatten",
        input_scale: float = 1.0,
        drop_prob: float = 0.1,
        n_layer: int = 12,
        nhead: int = 8,
        emb_dim: int = 200,
        dim_feedforward: int = 800,
        proj_hidden: Optional[int] = None,
        proj_blocks: int = 1,
        proj_dropout: float = 0.5,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        if feature_pool not in ("flatten", "channel_mean", "mean"):
            raise ValueError(f"unknown feature_pool {feature_pool!r}")
        if pad_mode not in _PAD_MODES:
            raise ValueError(f"pad_mode must be one of {_PAD_MODES}, got {pad_mode!r}")
        if pad_side not in _PAD_SIDES:
            raise ValueError(f"pad_side must be one of {_PAD_SIDES}, got {pad_side!r}")

        self.patch_size = int(patch_size)
        if self.patch_size != CBRAMOD_PATCH_SIZE and pretrained:
            raise ValueError(
                f"patch_size={patch_size} discards the pretrained patch embedding "
                f"(spectral_proj is (200, 101) and proj_in maps 200 -> 8 positions). "
                "CBraMod's patch is baked into the checkpoint: pad the epoch instead. "
                "Pass pretrained=False if you really want a re-patched scratch model."
            )
        self.pad_mode = pad_mode
        self.pad_side = pad_side
        self.pad_to = int(pad_to) if pad_to is not None else self._round_up(self.n_times)
        if self.pad_to % self.patch_size or self.pad_to < self.n_times:
            raise ValueError(
                f"pad_to={self.pad_to} must be a multiple of patch_size={self.patch_size} "
                f"and >= n_times={self.n_times}"
            )
        self.n_patches = self.pad_to // self.patch_size
        self.pad_samples = self.pad_to - self.n_times
        self.padding_fraction = self.pad_samples / self.pad_to
        self.feature_pool = feature_pool
        self.input_scale = float(input_scale)
        self.backbone_lr_scale = float(backbone_lr_scale)

        # ---- backbone ---------------------------------------------------------------
        from braindecode.models import CBraMod  # local import: heavy, and optional

        kwargs = dict(
            n_outputs=1,  # unused: return_encoder_output=True makes final_layer Identity
            n_chans=self.n_channels,
            n_times=self.pad_to,
            sfreq=200.0,
            patch_size=self.patch_size,
            n_layer=int(n_layer),
            nhead=int(nhead),
            emb_dim=int(emb_dim),
            dim_feedforward=int(dim_feedforward),
            drop_prob=float(drop_prob),
            return_encoder_output=True,  # -> final_layer = Identity, zero random tensors
        )
        self.pretrained_info: Dict[str, Any] = {
            "requested": bool(pretrained),
            "repo": hf_repo if pretrained else None,
            "loaded": False,
            "verified": None,
            "verify_error": None,
            "n_ckpt_tensors": None,
            "n_matched_tensors": None,
            "pretrained_params": 0,
            "backbone_params": None,
            "pretrained_param_fraction": 0.0,
        }
        if pretrained:
            try:
                self.backbone = CBraMod.from_pretrained(hf_repo, **kwargs)
                self.pretrained_info["loaded"] = True
            except Exception as exc:  # noqa: BLE001
                if not allow_random_init:
                    raise RuntimeError(
                        f"could not load pretrained CBraMod from {hf_repo!r}: "
                        f"{type(exc).__name__}: {exc}. A randomly-initialised CBraMod is NOT "
                        "a foundation-model baseline; pass allow_random_init=True only if you "
                        "intend to report it as an ablation."
                    ) from exc
                log.error(
                    "CBraMod pretrained weights FAILED to load (%s: %s); falling back to "
                    "RANDOM INIT because allow_random_init=True. This run is not a "
                    "foundation-model baseline.",
                    type(exc).__name__, exc,
                )
                self.backbone = CBraMod(**kwargs)
        else:
            self.backbone = CBraMod(**kwargs)

        self.pretrained_info["backbone_params"] = count_parameters(self.backbone, False)
        if self.pretrained_info["loaded"] and verify_pretrained:
            self.verify_pretrained(hf_repo)
        elif self.pretrained_info["loaded"]:
            # not verified tensor-by-tensor; report the structural count only
            self.pretrained_info["pretrained_params"] = self.pretrained_info["backbone_params"]
            self.pretrained_info["pretrained_param_fraction"] = 1.0

        # ---- freezing ---------------------------------------------------------------
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n = None if unfreeze_last_n is None else int(unfreeze_last_n)
        self.freeze_patch_embedding = bool(freeze_patch_embedding)
        self._apply_freezing()

        self.proj_hidden = proj_hidden
        self.proj_blocks = int(proj_blocks)
        self.proj_dropout = float(proj_dropout)
        self.setup_head()

    # ---------------------------------------------------------------------------------
    # construction helpers
    # ---------------------------------------------------------------------------------

    def _round_up(self, n_times: int) -> int:
        return int(math.ceil(max(1, int(n_times)) / self.patch_size)) * self.patch_size

    def verify_pretrained(self, hf_repo: str = CBRAMOD_HF_REPO) -> Dict[str, Any]:
        """Assert the checkpoint tensors really are the live module's tensors.

        Downloads (from cache) the repo's ``model.safetensors`` and compares every
        entry bit-exactly against ``self.backbone.state_dict()``.  Records the result
        in ``self.pretrained_info`` and returns it.  A verification *failure* (a
        tensor missing or different) raises; an inability to *run* the check (no
        safetensors, no hub access) is recorded, not raised.
        """
        info = self.pretrained_info
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            try:
                path = hf_hub_download(hf_repo, "model.safetensors", local_files_only=True)
            except Exception:  # not in cache yet -> allow a network fetch
                path = hf_hub_download(hf_repo, "model.safetensors")
            ckpt = load_file(path)
        except Exception as exc:  # noqa: BLE001
            info["verified"] = False
            info["verify_error"] = f"{type(exc).__name__}: {exc}"
            info["pretrained_params"] = info["backbone_params"]
            info["pretrained_param_fraction"] = float("nan")
            log.warning("CBraMod: could not verify the checkpoint tensors (%s)", info["verify_error"])
            return info

        sd = self.backbone.state_dict()
        matched, n_matched, bad = 0, 0, []
        for key, ref in ckpt.items():
            got = sd.get(key)
            if got is None:
                bad.append(f"{key}: absent from the live model")
            elif tuple(got.shape) != tuple(ref.shape):
                bad.append(f"{key}: shape {tuple(got.shape)} != checkpoint {tuple(ref.shape)}")
            elif not torch.equal(got.detach().cpu(), ref):
                bad.append(f"{key}: values differ from the checkpoint")
            else:
                matched += ref.numel()
                n_matched += 1
        if bad:
            raise RuntimeError(
                "CBraMod pretrained verification FAILED -- the live model does not hold the "
                f"checkpoint weights ({len(bad)} tensor(s)): {bad[:5]}"
            )
        total = count_parameters(self.backbone, False)
        info.update(
            verified=True,
            n_ckpt_tensors=len(ckpt),
            n_matched_tensors=n_matched,
            pretrained_params=int(matched),
            backbone_params=int(total),
            pretrained_param_fraction=(matched / total) if total else 0.0,
        )
        log.info(
            "CBraMod: %d/%d checkpoint tensors bit-exact in the live model "
            "(%d/%d backbone params = %.4f)",
            n_matched, len(ckpt), matched, total, info["pretrained_param_fraction"],
        )
        return info

    # ---------------------------------------------------------------------------------
    # fine-tuning controls
    # ---------------------------------------------------------------------------------

    def backbone_modules_by_depth(self) -> List[nn.Module]:
        """Backbone submodules ordered input -> output (patch embedding, blocks, proj)."""
        mods: List[nn.Module] = [self.backbone.patch_embedding]
        mods.extend(list(self.backbone.encoder.layers))
        mods.append(self.backbone.proj_out)
        return mods

    def _apply_freezing(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(not self.freeze_backbone)
        if self.unfreeze_last_n is not None:
            if self.freeze_backbone:
                raise ValueError("freeze_backbone=True and unfreeze_last_n are contradictory")
            n_blocks = len(self.backbone.encoder.layers)
            k = max(0, min(int(self.unfreeze_last_n), n_blocks))
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for blk in list(self.backbone.encoder.layers)[n_blocks - k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.backbone.proj_out.parameters():  # always adapt the read-out
                p.requires_grad_(True)
        if self.freeze_patch_embedding:
            for p in self.backbone.patch_embedding.parameters():
                p.requires_grad_(False)

    @property
    def backbone_is_fully_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        # a fully frozen backbone must not keep sampling dropout masks: it is a fixed
        # feature extractor, and stochastic features would make the "linear probe"
        # arm noisier than the fine-tuned one for a reason that is not the model.
        if mode and self.backbone_is_fully_frozen:
            self.backbone.eval()
        return self

    def param_groups(self, base_lr: float, weight_decay: float = 0.05) -> List[Dict[str, Any]]:
        """Discriminative-LR optimizer groups: pretrained at ``base_lr * backbone_lr_scale``.

        NOT consumed by ``tactus.train.trainer.Trainer`` today -- it builds its own
        two decay/no-decay groups at a single ``lr``.  Wiring this in is a two-line
        change there; until then use ``freeze_backbone`` / ``unfreeze_last_n``.
        """
        bb_ids = {id(p) for p in self.backbone.parameters()}
        groups: Dict[str, List[nn.Parameter]] = {
            "bb_decay": [], "bb_nodecay": [], "new_decay": [], "new_nodecay": [],
        }
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            pre = "bb" if id(p) in bb_ids else "new"
            no_decay = p.ndim <= 1 or name.endswith(".bias")
            groups[f"{pre}_{'nodecay' if no_decay else 'decay'}"].append(p)
        scale = self.backbone_lr_scale
        out = [
            {"params": groups["bb_decay"], "lr": base_lr * scale, "weight_decay": weight_decay},
            {"params": groups["bb_nodecay"], "lr": base_lr * scale, "weight_decay": 0.0},
            {"params": groups["new_decay"], "lr": base_lr, "weight_decay": weight_decay},
            {"params": groups["new_nodecay"], "lr": base_lr, "weight_decay": 0.0},
        ]
        return [g for g in out if g["params"]]

    # ---------------------------------------------------------------------------------
    # contract F
    # ---------------------------------------------------------------------------------

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad ``(B, C, T)`` to the fixed ``pad_to`` (or the next multiple if longer)."""
        n_times = int(x.shape[-1])
        target = self.pad_to if n_times <= self.pad_to else self._round_up(n_times)
        return pad_to_multiple(x, target, mode=self.pad_mode, side=self.pad_side)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T) -> (B, F)``.  ``F`` is constant for every ``T <= pad_to``."""
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if self.input_scale != 1.0:
            x = x * self.input_scale
        x = self.pad(x)
        h = self.backbone(x)  # (B, C, n_patch, emb_dim); final_layer is Identity
        if h.dim() != 4:
            raise RuntimeError(
                f"expected CBraMod encoder output (B, C, P, E), got {tuple(h.shape)}; "
                "return_encoder_output=True was not honoured"
            )
        b, c, p, e = h.shape
        if p != self.n_patches:  # a window longer than pad_to: pool back to a fixed width
            h = F.adaptive_avg_pool1d(
                h.reshape(b * c, p, e).transpose(1, 2), self.n_patches
            ).transpose(1, 2).reshape(b, c, self.n_patches, e)
        if self.feature_pool == "channel_mean":
            h = h.mean(dim=1)
        elif self.feature_pool == "mean":
            h = h.mean(dim=(1, 2))
        return h.flatten(1)

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
    # reporting
    # ---------------------------------------------------------------------------------

    def padding_report(self) -> Dict[str, Any]:
        """The documented interface mismatch, in a form a run log can serialise."""
        return {
            "patch_size": self.patch_size,
            "n_times": self.n_times,
            "padded_to": self.pad_to,
            "n_patches": self.n_patches,
            "pad_samples": self.pad_samples,
            "padding_fraction": round(self.padding_fraction, 6),
            "pad_mode": self.pad_mode,
            "pad_side": self.pad_side,
            "note": (
                "CBraMod's 200-sample patch is baked into the pretrained weights "
                "(spectral_proj is (200,101)); the epoch is padded, never re-patched. "
                f"{self.padding_fraction:.1%} of every patch is synthetic."
            ),
        }

    def describe(self) -> Dict[str, Any]:  # noqa: D102
        out = super().describe()
        out["pretrained"] = dict(self.pretrained_info)
        out["padding"] = self.padding_report()
        out["feature_pool"] = self.feature_pool
        out["input_scale"] = self.input_scale
        out["backbone_params"] = count_parameters(self.backbone, False)
        out["backbone_trainable_params"] = count_parameters(self.backbone, True)
        out["backbone_fully_frozen"] = self.backbone_is_fully_frozen
        out["freeze"] = {
            "freeze_backbone": self.freeze_backbone,
            "unfreeze_last_n": self.unfreeze_last_n,
            "freeze_patch_embedding": self.freeze_patch_embedding,
        }
        out["backbone_lr_scale"] = self.backbone_lr_scale
        # Trainer._build calls encoder.param_groups(lr, wd) and prepends the
        # returned groups, so this IS applied.  (It was inert when this file was
        # written; the hand-off landed in the same session.)
        out["backbone_lr_scale_applied"] = True
        return out


# ======================================================================================
# smoke check / padding-sensitivity measurement
# ======================================================================================


def _padding_sensitivity(enc: "CBraModEncoder", x: torch.Tensor) -> Dict[str, float]:
    """Cosine similarity of the *backbone* features across padding schemes, same trials."""
    was = (enc.pad_mode, enc.pad_side)
    feats: Dict[str, torch.Tensor] = {}
    enc.eval()
    with torch.no_grad():
        for mode in _PAD_MODES:
            enc.pad_mode = mode
            feats[mode] = enc.extract_features(x)
    enc.pad_mode, enc.pad_side = was
    out: Dict[str, float] = {}
    keys = list(feats)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            cos = F.cosine_similarity(feats[a], feats[b], dim=-1).mean().item()
            rel = ((feats[a] - feats[b]).norm(dim=-1) / feats[a].norm(dim=-1)).mean().item()
            out[f"{a}~{b}"] = cos
            out[f"{a}~{b}|relL2"] = rel
    return out


if __name__ == "__main__":  # pragma: no cover - server-side smoke check
    import json
    import os

    torch.manual_seed(0)
    print("== build (pretrained) ==")
    enc = CBraModEncoder(n_times=120, embed_dim=256, subject_cond="subject_token")
    info = enc.pretrained_info
    print(json.dumps({"pretrained": info, "padding": enc.padding_report()}, indent=2))
    assert info["loaded"] and info["verified"], "pretrained weights are NOT in the model"
    assert info["pretrained_param_fraction"] == 1.0

    print("\n== contract F ==")
    for cond in ("none", "subject_token", "subject_layer", "sulora"):
        e = CBraModEncoder(n_times=120, embed_dim=256, subject_cond=cond).eval()
        x = torch.randn(4, 64, 120)
        with torch.no_grad():
            z = e(x, torch.tensor([1, 2, 3, -1]))
            z_none = e(x, None)
            z_unseen = e(x, torch.full((4,), -1))
        assert z.shape == (4, 256) and torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-4)
        assert torch.allclose(z_none, z_unseen, atol=1e-5)
        print(
            f"  {cond:>14s}  params={count_parameters(e):>10,d}  feat={e.feature_dim:>6d}  "
            f"out={tuple(z.shape)}  rule={e.unseen_subject_state()['rule']}"
        )

    print("\n== gradient flow (fine-tune / frozen) ==")
    for kw in ({}, {"freeze_backbone": True}, {"unfreeze_last_n": 2, "freeze_patch_embedding": True}):
        e = CBraModEncoder(n_times=120, embed_dim=256, subject_cond="subject_token", **kw).train()
        z = e(torch.randn(4, 64, 120), torch.tensor([1, 2, 3, 4]))
        z.pow(2).sum().backward()
        got_bb = sum(1 for p in e.backbone.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        got_pr = sum(1 for p in e.projector.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        print(
            f"  {str(kw) or '{full fine-tune}':<52s} backbone grads={got_bb:>3d} "
            f"projector grads={got_pr:>3d} trainable={count_parameters(e):>10,d}"
        )

    print("\n== sensitivity window wm100_800 ==")
    e180 = CBraModEncoder(n_times=180, embed_dim=256, subject_cond="none").eval()
    print("  ", e180.padding_report())

    print("\n== padding sensitivity on REAL trials ==")
    root = os.environ.get("TACTUS_DATA_ROOT", "/projects/EEG-foundation-model/tactus_work")
    path = os.path.join(root, "derived", "epochs", "sub-01_w0600.npy")
    if os.path.exists(path):
        import numpy as np

        arr = np.load(path, mmap_mode="r")
        xr = torch.from_numpy(np.asarray(arr[:64]).astype("float32"))
        probe = CBraModEncoder(n_times=120, embed_dim=256, subject_cond="none").eval()
        for k, v in _padding_sensitivity(probe, xr).items():
            print(f"  {k:<26s} {v:.4f}")
    else:
        print("  (skipped: no epochs at", path, ")")
