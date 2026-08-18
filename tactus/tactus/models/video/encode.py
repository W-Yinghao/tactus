#!/usr/bin/env python
"""Build the frozen video-embedding cache (contract C).

Output: ``data/derived/video_emb/{model_tag}.npz`` with

* ``cond_emb`` ``(360, D)`` float32, L2-normalized, indexed by
  ``condition_id = (video_id - 1) * 4 + orientation``
* ``base_emb`` ``(90, D)`` float32, L2-normalized, indexed by ``video_id - 1``
  (original orientation only)
* ``frame_emb`` ``(360, n_frames, D)`` float32, L2-normalized per row, written
  only with ``--save-frame-emb`` and only for the ``image_clip`` family
* ``meta`` -- a JSON string with model name, pooling, ``n_frames`` and provenance

**Provenance is part of the contract.** A cache built on CUDA with ``--fp16`` and
one built on CPU in float32 are not bit-identical: measured on siglip2-base, the
two agree at cosine 0.9999982 (min 0.9999849) and give the same top-1 nearest
video for 89 of 90 stimuli. That is numerics, not a defect, but it means a
``frame_emb`` array must be used with the ``cond_emb`` from *its own* file rather
than with the canonical one -- inside one file the pooled vector reproduces the
frame mean at cosine 0.99999985.

Why all 360 conditions and not just 90: a mirrored touch video is a *visually
different stimulus*, and the orientation-equivariance analysis (BLUEPRINT_v2
contribution 2) needs the encoder's own answer to "how different".  Embedding the
90 originals and reusing them for the flips would build the conclusion into the data.

Encoder families
----------------
``image_clip``
    Frame-wise ``get_image_features``, each frame L2-normalized, averaged, then
    re-normalized (SigLIP2, SigLIP, CLIP).  Order-blind -- a bag of frames.
``xclip``
    ``get_video_features`` on the whole clip (X-CLIP); temporal.
``video_ssl``
    Mean of the tubelet/patch tokens of ``last_hidden_state`` (VideoMAE v1,
    V-JEPA 2, InternVideo2); temporal.
``videomae_v2``
    OpenGVLab VideoMAEv2, whose remote modelling code cannot be loaded by
    ``AutoModel.from_pretrained`` under transformers 5 -- see ``_load_videomae_v2``.
    ``extract_features`` on a ``(B, C, T, H, W)`` tensor; temporal.

Whether an encoder is *actually* temporal is an empirical question, not a family
label: use ``--frame-order`` to measure it.  ``image_clip`` is order-blind by
construction (a mean over per-frame embeddings is permutation-invariant), so for that
family ``cos(native, shuffled) == 1`` is arithmetic, not a fact about the backbone.

Resumability
------------
Each condition is written to ``{out}/_cache/{tag}/cond_{id:03d}.npy`` as it is
computed, and an interrupted run resumes by skipping conditions already on disk.
The final ``.npz`` is assembled from the shards.  ``--force`` recomputes everything.

Usage
-----
::

    python -m tactus.models.video.encode --list-models
    python -m tactus.models.video.encode --stim-root ds005662 --model siglip2-base
    python -m tactus.models.video.encode --stim-root ds005662 --model vjepa2-vitl --device cuda
    python -m tactus.models.video.encode --stim-root ds005662 --dry-run       # validate paths only
    python -m tactus.models.video.encode --stim-root ds005662 --verify-flips  # check the axis convention
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "ORIENTATION_NAMES",
    "condition_id",
    "parse_orientation",
    "parse_video_id",
    "discover_orientation_dirs",
    "read_video_frames",
    "build_cache",
    "FRAME_ORDERS",
    "DEFAULT_FRAME_ORDER_SEED",
    "frame_permutation",
    "reorder_frames",
    "repair_and_verify_weights",
]

N_VIDEOS = 90
N_ORIENTATIONS = 4
N_CONDITIONS = N_VIDEOS * N_ORIENTATIONS

#: orientation code -> canonical name (contract A: 0=original, 1=horflip, 2=vertflip, 3=horvertflip)
ORIENTATION_NAMES: Dict[int, str] = {0: "original", 1: "horflip", 2: "vertflip", 3: "horvertflip"}

#: Filename/dirname tokens, longest first.  ``horvertflip`` MUST be tested before
#: ``horflip``/``vertflip`` -- it contains both as substrings.
_ORIENTATION_TOKENS: Tuple[Tuple[str, int], ...] = (
    ("horvertflip", 3),
    ("horflip", 1),
    ("vertflip", 2),
)


# ======================================================================================
# model registry
# ======================================================================================


@dataclass(frozen=True)
class ModelSpec:
    """A frozen video/image encoder that can produce a clip embedding."""

    tag: str
    hf_id: str
    family: str  # "image_clip" | "xclip" | "video_ssl"
    n_frames: int
    pooling: str
    trust_remote_code: bool = False
    notes: str = ""

    def describe(self) -> str:  # noqa: D102
        flag = " [trust_remote_code]" if self.trust_remote_code else ""
        return (
            f"{self.tag:<20s} {self.family:<11s} frames={self.n_frames:<3d} "
            f"{self.hf_id}{flag}\n{'':<20s} {self.pooling}"
            + (f"\n{'':<20s} note: {self.notes}" if self.notes else "")
        )


def _spec(**kw: Any) -> ModelSpec:
    return ModelSpec(**kw)


#: The Phase-0 encoder census (BLUEPRINT_v2 section 4.1) spans three families:
#: text-aligned (SigLIP2 / X-CLIP / InternVideo2), SSL (VideoMAE v2 / V-JEPA 2), and
#: tactile-semantic (UniTouch / TVL -- not loadable through ``transformers``, see notes).
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    s.tag: s
    for s in [
        _spec(
            tag="siglip2-base",
            hf_id="google/siglip2-base-patch16-224",
            family="image_clip",
            n_frames=15,
            pooling="per-frame get_image_features -> L2 -> mean -> L2",
            notes="default; ~15 frames is the full 600 ms clip at 25 fps",
        ),
        _spec(
            tag="siglip2-so400m",
            hf_id="google/siglip2-so400m-patch14-384",
            family="image_clip",
            n_frames=15,
            pooling="per-frame get_image_features -> L2 -> mean -> L2",
            notes="large; use if siglip2-base collapses in the census",
        ),
        _spec(
            tag="siglip-base",
            hf_id="google/siglip-base-patch16-224",
            family="image_clip",
            n_frames=15,
            pooling="per-frame get_image_features -> L2 -> mean -> L2",
        ),
        _spec(
            tag="clip-vit-l14",
            hf_id="openai/clip-vit-large-patch14",
            family="image_clip",
            n_frames=15,
            pooling="per-frame get_image_features -> L2 -> mean -> L2",
            notes="the NICE/ATM reference encoder; keep for comparability with prior work",
        ),
        _spec(
            tag="xclip-base-32",
            hf_id="microsoft/xclip-base-patch32",
            family="xclip",
            n_frames=8,
            pooling="get_video_features (temporal cross-frame attention) -> L2",
            notes="expects exactly 8 frames",
        ),
        _spec(
            tag="xclip-base-16",
            hf_id="microsoft/xclip-base-patch16",
            family="xclip",
            n_frames=8,
            pooling="get_video_features -> L2",
        ),
        _spec(
            tag="videomae-v2-base",
            hf_id="OpenGVLab/VideoMAEv2-Base",
            family="videomae_v2",
            n_frames=16,
            pooling="extract_features = fc_norm(mean of tubelet tokens) -> L2",
            trust_remote_code=True,
            notes="custom modelling code written against transformers 4.x; "
                  "AutoModel.from_pretrained CANNOT load it under transformers 5 "
                  "(the remote class has no all_tied_weights_keys, and its sinusoidal "
                  "pos_embed buffer is left on the meta device). Loaded here by direct "
                  "instantiation + load_state_dict instead; needs `easydict`.",
        ),
        _spec(
            tag="videomae-base-k400",
            hf_id="MCG-NJU/videomae-base-finetuned-kinetics",
            family="video_ssl",
            n_frames=16,
            pooling="mean of tubelet tokens -> L2",
            notes="VideoMAE v1; a robust fallback if the v2 remote code breaks",
        ),
        _spec(
            tag="vjepa2-vitl",
            hf_id="facebook/vjepa2-vitl-fpc64-256",
            family="video_ssl",
            n_frames=64,
            pooling="mean of patch tokens -> L2",
            notes="needs transformers>=4.52; 64 frames means heavy temporal padding "
            "of a 15-frame clip -- record that in the census",
        ),
        _spec(
            tag="internvideo2-s2-1b",
            hf_id="OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
            family="video_ssl",
            n_frames=4,
            pooling="mean of patch tokens -> L2",
            trust_remote_code=True,
            notes="the f4 variant from the blueprint; may need extra deps (flash-attn)",
        ),
    ]
}

_UNSUPPORTED_NOTE = """
Not in the registry because they do not load through `transformers.AutoModel`:
  UniTouch (CVPR'24) ImageBind vision tower, TVL (ICML'24) tactile-adjective scorer.
Both are first-person tactile-sensor models; on third-person video they are expected to
carry the material axis only. Add them via --model-id/--family only if you have a
transformers-compatible export, otherwise embed them out-of-band and drop a matching
.npz into the cache directory.
""".strip()


# ======================================================================================
# stimulus discovery
# ======================================================================================


def condition_id(video_id: int, orientation: int) -> int:
    """``(video_id - 1) * 4 + orientation`` (contract A)."""
    if not 1 <= int(video_id) <= N_VIDEOS:
        raise ValueError(f"video_id must be in 1..{N_VIDEOS}, got {video_id}")
    if not 0 <= int(orientation) < N_ORIENTATIONS:
        raise ValueError(f"orientation must be in 0..3, got {orientation}")
    return (int(video_id) - 1) * N_ORIENTATIONS + int(orientation)


def parse_orientation(name: str) -> int:
    """Orientation code from a file or directory name.

    ``horvertflip`` is matched before ``horflip``/``vertflip``; a name with no flip
    token is the original orientation (0).
    """
    low = str(name).lower()
    for token, code in _ORIENTATION_TOKENS:
        if token in low:
            return code
    return 0


def parse_video_id(filename: str) -> int:
    """Base video id = the leading integer of the basename, before the first period."""
    m = re.match(r"^(\d+)\.", Path(filename).name)
    if not m:
        raise ValueError(f"cannot parse a leading video id from {filename!r}")
    return int(m.group(1))


def resolve_stimuli_root(path: Path) -> Path:
    """Accept the dataset root, ``code/experiment_files``, or the ``stimuli`` dir itself."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    candidates = [
        path,
        path / "stimuli",
        path / "experiment_files" / "stimuli",
        path / "code" / "experiment_files" / "stimuli",
    ]
    for cand in candidates:
        if cand.is_dir() and any(
            d.is_dir() and "600ms" in d.name.lower() and "target" not in d.name.lower()
            for d in cand.iterdir()
        ):
            return cand
    raise FileNotFoundError(
        f"could not find a stimuli directory under {path}; looked in "
        f"{[str(c) for c in candidates]}. Expected sub-directories named like "
        "'Videos_short_600ms', 'Videos_short_600ms_horflip', ..."
    )


def discover_orientation_dirs(stim_root: Path) -> Dict[int, Path]:
    """Map orientation code -> directory of the 90 non-target videos in that orientation."""
    found: Dict[int, Path] = {}
    for d in sorted(Path(stim_root).iterdir()):
        if not d.is_dir():
            continue
        low = d.name.lower()
        if "600ms" not in low or "target" in low:
            continue
        code = parse_orientation(low)
        if code in found:
            raise RuntimeError(
                f"two directories map to orientation {code} ({ORIENTATION_NAMES[code]}): "
                f"{found[code].name} and {d.name}"
            )
        found[code] = d
    missing = sorted(set(ORIENTATION_NAMES) - set(found))
    if missing:
        raise FileNotFoundError(
            f"missing orientation directories for {[ORIENTATION_NAMES[m] for m in missing]} "
            f"under {stim_root}; found {[d.name for d in found.values()]}. "
            "Use --synthesize-flips to derive them from the originals instead."
        )
    return found


def index_videos(orient_dirs: Dict[int, Path], strict: bool = True) -> Dict[int, Path]:
    """Map ``condition_id -> mp4 path``, validating the 90 x 4 grid."""
    index: Dict[int, Path] = {}
    for code, d in sorted(orient_dirs.items()):
        files = sorted(d.glob("*.mp4"))
        ids: List[int] = []
        for f in files:
            try:
                vid = parse_video_id(f.name)
            except ValueError:
                continue
            if parse_orientation(f.name) != code:
                raise RuntimeError(
                    f"{f} sits in the {ORIENTATION_NAMES[code]} directory but its filename "
                    f"parses as {ORIENTATION_NAMES[parse_orientation(f.name)]}"
                )
            ids.append(vid)
            index[condition_id(vid, code)] = f
        dupes = {v for v in ids if ids.count(v) > 1}
        if dupes:
            raise RuntimeError(f"duplicate video ids {sorted(dupes)} in {d}")
        if strict and len(ids) != N_VIDEOS:
            raise RuntimeError(f"expected {N_VIDEOS} videos in {d}, found {len(ids)}")
    if strict and len(index) != N_CONDITIONS:
        raise RuntimeError(f"expected {N_CONDITIONS} conditions, indexed {len(index)}")
    return index


# ======================================================================================
# decoding
# ======================================================================================


def read_video_frames(path: Path, n_frames: Optional[int] = None) -> Tuple[List[np.ndarray], int]:
    """Decode ``path`` to a list of RGB ``uint8`` frames, uniformly resampled to ``n_frames``.

    Returns ``(frames, n_decoded)`` so the caller can record how much temporal padding
    a 64-frame model needed for a 15-frame clip.
    """
    import cv2  # local import: only the decoding path needs OpenCV

    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}; is the OpenCV/ffmpeg build able to read mp4?")
    n_decoded = len(frames)
    if n_frames is None or n_frames == n_decoded:
        return frames, n_decoded
    # uniform resample (repeats frames when the model wants more than the clip has)
    idx = np.linspace(0, n_decoded - 1, num=int(n_frames))
    return [frames[int(round(i))] for i in idx], n_decoded


def flip_frames(frames: Sequence[np.ndarray], orientation: int) -> List[np.ndarray]:
    """Derive a flipped clip from the original frames.

    Convention (verify with ``--verify-flips`` against the shipped files):
    ``horflip`` mirrors left-right (image axis 1), ``vertflip`` mirrors up-down
    (axis 0), ``horvertflip`` does both.
    """
    if orientation == 0:
        return list(frames)
    if orientation == 1:
        return [np.ascontiguousarray(f[:, ::-1]) for f in frames]
    if orientation == 2:
        return [np.ascontiguousarray(f[::-1, :]) for f in frames]
    if orientation == 3:
        return [np.ascontiguousarray(f[::-1, ::-1]) for f in frames]
    raise ValueError(f"bad orientation {orientation}")


# ======================================================================================
# frame-order control
# ======================================================================================

#: Temporal manipulations applied *within* a clip, after the uniform resample to
#: ``n_frames`` and therefore to exactly the frames the encoder sees.
FRAME_ORDERS: Tuple[str, ...] = ("native", "shuffled", "reversed")

#: Default seed for ``--frame-order shuffled``.  Pinned so the permutation is a
#: property of the cache tag, not of when the job happened to run.
DEFAULT_FRAME_ORDER_SEED = 20260816


def frame_permutation(order: str, n_frames: int, seed: int, video_key: int) -> np.ndarray:
    """Deterministic within-clip frame permutation for base video ``video_key``.

    * ``native``   -- identity.
    * ``reversed`` -- exact time reversal.  The sharp test: an approaching hand
      played backwards is a retreating hand, so any encoder with directional
      sensitivity must move.
    * ``shuffled`` -- a random permutation drawn from
      ``np.random.default_rng([seed, video_key, n_frames])``, rejecting the identity.

    Two deliberate choices:

    *Per video, not one global permutation.*  A single fixed permutation applied to
    every clip does not destroy temporal order, it re-codes it into another fixed
    order that a downstream model can learn just as well.  A fresh draw per video
    destroys it.

    *Per base video, not per condition.*  ``video_key`` is ``condition_id // 4``, so
    the four orientations of one video share a scramble.  The orientation structure
    of the cache is then bit-for-bit the structure of the native cache and the only
    thing the control changes is time.  Keying on ``condition_id`` instead would also
    untie the four flips of a video from each other, and a training arm could not
    tell the two effects apart.

    The permutation never crosses clip boundaries and never touches which video a
    ``condition_id`` maps to -- only the order of that clip's own frames.
    """
    n = int(n_frames)
    idx = np.arange(n, dtype=np.int16)
    if order == "native":
        return idx
    if order == "reversed":
        return idx[::-1].copy()
    if order != "shuffled":
        raise ValueError(f"unknown frame order {order!r}; expected one of {FRAME_ORDERS}")
    if n < 2:
        return idx
    rng = np.random.default_rng([int(seed), int(video_key), n])
    for _ in range(64):
        perm = rng.permutation(idx)
        if not np.array_equal(perm, idx):  # reject the identity draw
            return perm.astype(np.int16)
    raise RuntimeError(
        f"could not draw a non-identity permutation for video_key={video_key}, n_frames={n}"
    )


def reorder_frames(frames: Sequence[np.ndarray], perm: Sequence[int]) -> List[np.ndarray]:
    """Apply ``perm`` to one clip's frames.  ``len(perm)`` must equal ``len(frames)``."""
    if len(perm) != len(frames):
        raise ValueError(f"permutation of length {len(perm)} for a {len(frames)}-frame clip")
    return [frames[int(i)] for i in perm]


# ======================================================================================
# models
# ======================================================================================


#: Checkpoint parameter names that ``transformers`` renamed without shipping a
#: conversion map, keyed by the *suffix* of the live parameter name.
#:
#: VideoMAE: up to transformers 4.x ``VideoMAESelfAttention`` held the attention bias
#: as two bare ``nn.Parameter``\s -- ``q_bias`` and ``v_bias`` -- with the key bias
#: hard-wired to zero (``k_bias = torch.zeros_like(v_bias)``).  transformers 5.x
#: rewrote the module to plain ``nn.Linear(..., bias=config.qkv_bias)``, so the live
#: model now wants ``query.bias``/``key.bias``/``value.bias`` and every published
#: VideoMAE checkpoint (they all use the old names) silently loses its q and v bias.
#: ``VideoMAEPreTrainedModel`` carries no ``_checkpoint_conversion_mapping``, so
#: nothing puts them back.  These aliases do.
_CHECKPOINT_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "attention.attention.query.bias": ("attention.attention.q_bias",),
    "attention.attention.value.bias": ("attention.attention.v_bias",),
}

#: Live parameters that legitimately have no checkpoint counterpart because the
#: original implementation held them at a constant.  Value = the constant.
_CHECKPOINT_CONSTANTS: Dict[str, float] = {
    "attention.attention.key.bias": 0.0,  # old VideoMAE: k_bias := zeros, never trained
}


def _raw_checkpoint_state_dict(spec: ModelSpec) -> Dict[str, "Any"]:
    """The checkpoint exactly as published, before any ``transformers`` remapping."""
    import torch
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            spec.hf_id,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.index.json"],
        )
    )
    sd: Dict[str, Any] = {}
    shards = sorted(local.glob("*.safetensors"))
    if shards:
        from safetensors.torch import load_file

        for shard in shards:
            sd.update(load_file(str(shard)))
        return sd
    for shard in sorted(local.glob("*.bin")):
        sd.update(torch.load(str(shard), map_location="cpu", weights_only=True))
    if not sd:
        raise RuntimeError(f"no .safetensors or .bin weights found in {local}")
    return sd


def repair_and_verify_weights(model, spec: ModelSpec, repair: bool = True) -> Dict[str, Any]:
    """Compare every live tensor against the published checkpoint; repair known renames.

    Returns a report dict.  ``status == "ok"`` means every live parameter is either
    byte-identical to a checkpoint tensor, restored from a renamed one, or a documented
    constant.  Anything else is listed under ``unmatched`` and the encoder must not be
    trusted to represent what its checkpoint says it represents.
    """
    import torch

    report: Dict[str, Any] = {
        "hf_id": spec.hf_id,
        "checked": 0,
        "matched": 0,
        "repaired": [],
        "constants": [],
        "unmatched": [],
        "repair_enabled": bool(repair),
    }
    try:
        ckpt = _raw_checkpoint_state_dict(spec)
    except Exception as exc:  # noqa: BLE001 - remote-code models may not be plain checkpoints
        report["status"] = "unverified"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    prefix = getattr(model, "base_model_prefix", "") or ""
    # A task checkpoint (e.g. ...ForVideoClassification) prefixes the backbone; strip it.
    flat: Dict[str, Any] = {}
    for k, v in ckpt.items():
        flat.setdefault(k, v)
        if prefix and k.startswith(prefix + "."):
            flat.setdefault(k[len(prefix) + 1 :], v)

    live = dict(model.state_dict())
    with torch.no_grad():
        for name, tensor in live.items():
            report["checked"] += 1
            src = flat.get(name)
            if src is not None and tuple(src.shape) == tuple(tensor.shape):
                if torch.equal(tensor.detach().cpu().float(), src.detach().cpu().float()):
                    report["matched"] += 1
                else:
                    report["unmatched"].append({"param": name, "why": "value differs from checkpoint"})
                continue

            alias_src = None
            for suffix, aliases in _CHECKPOINT_KEY_ALIASES.items():
                if not name.endswith(suffix):
                    continue
                stem = name[: -len(suffix)]
                for alias in aliases:
                    cand = flat.get(stem + alias)
                    if cand is not None and tuple(cand.shape) == tuple(tensor.shape):
                        alias_src = (stem + alias, cand)
                        break
                if alias_src:
                    break
            if alias_src is not None:
                key, cand = alias_src
                if repair:
                    tensor.copy_(cand.to(dtype=tensor.dtype, device=tensor.device))
                    report["repaired"].append({"param": name, "from": key})
                else:
                    report["unmatched"].append({"param": name, "why": f"renamed in checkpoint as {key}"})
                continue

            const = next((c for s, c in _CHECKPOINT_CONSTANTS.items() if name.endswith(s)), None)
            if const is not None:
                if repair:
                    tensor.fill_(float(const))
                report["constants"].append({"param": name, "value": float(const)})
                continue

            report["unmatched"].append({"param": name, "why": "absent from checkpoint"})

    # re-verify the repaired parameters actually landed
    if repair and report["repaired"]:
        live = dict(model.state_dict())
        bad = [
            r["param"]
            for r in report["repaired"]
            if not torch.equal(
                live[r["param"]].detach().cpu().float(), flat[r["from"]].detach().cpu().float()
            )
        ]
        if bad:
            raise RuntimeError(f"weight repair did not take effect for {bad[:4]}")
        report["matched"] += len(report["repaired"])

    report["status"] = "ok" if not report["unmatched"] else "incomplete"
    return report


def _load_videomae_v2(spec: ModelSpec):
    """Load OpenGVLab VideoMAEv2 without going through ``from_pretrained``.

    The repo's remote modelling code targets transformers 4.x.  Under transformers 5
    ``AutoModel.from_pretrained`` dies twice over: the remote class predates
    ``PreTrainedModel.all_tied_weights_keys``, and even with that shimmed the
    meta-device init path leaves the *sinusoidal* ``pos_embed`` buffer -- which is
    computed in ``__init__`` and deliberately absent from the checkpoint -- unmaterialised,
    so the first forward raises ``Cannot copy out of meta tensor``.

    Instantiating the class directly builds every buffer for real, and the checkpoint
    then loads with zero missing and zero unexpected keys.
    """
    import torch  # noqa: F401  (imported for parity with the rest of the module)
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(spec.hf_id, trust_remote_code=True)
    cls = get_class_from_dynamic_module("modeling_videomaev2.VideoMAEv2", spec.hf_id)
    model = cls(cfg)
    local = Path(snapshot_download(spec.hf_id))
    shards = sorted(local.glob("*.safetensors"))
    sd: Dict[str, Any] = {}
    for shard in shards:
        sd.update(load_file(str(shard)))
    info = model.load_state_dict(sd, strict=False)
    missing = [k for k in info.missing_keys]
    unexpected = [k for k in info.unexpected_keys]
    if missing or unexpected:
        raise RuntimeError(
            f"VideoMAEv2 checkpoint did not load cleanly: {len(missing)} missing "
            f"{missing[:4]}, {len(unexpected)} unexpected {unexpected[:4]}"
        )
    return model


def _print_weight_report(rep: Dict[str, Any]) -> None:
    if rep.get("status") == "unverified":
        print(f"[encode] WEIGHTS unverified for {rep['hf_id']}: {rep.get('error')}")
        return
    print(
        f"[encode] weights: {rep['matched']}/{rep['checked']} tensors match the published "
        f"checkpoint; {len(rep['repaired'])} repaired from renamed keys; "
        f"{len(rep['constants'])} documented constants; {len(rep['unmatched'])} unmatched"
    )
    if rep["repaired"]:
        first = rep["repaired"][0]
        print(f"[encode]   repaired e.g. {first['from']} -> {first['param']}")
    for u in rep["unmatched"][:6]:
        print(f"[encode]   UNMATCHED {u['param']}: {u['why']}")
    if rep["unmatched"]:
        print("[encode]   WARNING: this encoder is NOT fully loaded; embeddings are partly untrained")


def load_model(spec: ModelSpec, device: str, fp16: bool = False, verify_weights: bool = True):
    """Load a frozen encoder and its processor.  Returns ``(model, processor)``.

    The weight report is attached as ``model.tactus_weight_report`` and copied into
    the cache meta, so a partly-initialised encoder can never quietly produce a cache.
    """
    from transformers import AutoModel

    kw: Dict[str, Any] = {"trust_remote_code": True} if spec.trust_remote_code else {}
    processor = _load_processor(spec)
    if spec.family == "videomae_v2":
        model = _load_videomae_v2(spec)
    else:
        model = AutoModel.from_pretrained(spec.hf_id, **kw)
    report: Dict[str, Any] = {"status": "skipped"}
    if verify_weights:
        report = repair_and_verify_weights(model, spec, repair=True)
        _print_weight_report(report)
    model.tactus_weight_report = report
    model = model.to(device).eval()
    if fp16 and str(device).startswith("cuda"):
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def _load_processor(spec: ModelSpec):
    """Try the processor classes in decreasing generality."""
    kw: Dict[str, Any] = {"trust_remote_code": True} if spec.trust_remote_code else {}
    errors: List[str] = []
    import transformers

    for name in ("AutoProcessor", "AutoVideoProcessor", "AutoImageProcessor", "AutoFeatureExtractor"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(spec.hf_id, **kw)
        except Exception as exc:  # noqa: BLE001 - we genuinely want to try the next one
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"no processor could be loaded for {spec.hf_id}:\n  " + "\n  ".join(errors))


def _video_inputs(processor, frames: Sequence[np.ndarray]):
    """Call a video processor across the several signatures transformers has used."""
    attempts = (
        lambda: processor(videos=[list(frames)], return_tensors="pt"),
        lambda: processor([list(frames)], return_tensors="pt"),
        lambda: processor(videos=list(frames), return_tensors="pt"),
        lambda: processor(images=list(frames), return_tensors="pt"),
    )
    errors: List[str] = []
    for call in attempts:
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("video processor rejected every call signature:\n  " + "\n  ".join(errors))


def _as_feature_tensor(out):
    """Unwrap whatever ``get_*_features`` returned into a plain tensor.

    transformers <5 returned a bare tensor; transformers >=5 returns a
    ``BaseModelOutputWithPooling``.  Calling ``.float()`` on the latter raises
    ``AttributeError``, so every call site goes through here.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "video_embeds", "text_embeds", "pooler_output",
                 "last_hidden_state"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise RuntimeError(
        f"cannot extract a feature tensor from {type(out).__name__}; "
        f"fields = {list(getattr(out, 'keys', lambda: [])())}"
    )


def embed_clip(frames: Sequence[np.ndarray], spec: ModelSpec, model, processor,
               device: str, return_frames: bool = False):
    """Embed one clip to a single L2-normalized vector ``(D,)``.

    With ``return_frames`` the result is ``(vec, per_frame)`` where ``per_frame``
    is ``(n_frames, D)`` L2-normalized, or ``None`` for the families that do not
    decompose that way.  Only ``image_clip`` does: its clip vector *is* the mean
    of per-frame vectors, which is why frame order is invisible to it -- a mean
    is permutation-invariant by arithmetic, not by any property of the encoder
    (DECISIONS D18).  ``xclip``, ``videomae_v2`` and ``video_ssl`` mix frames
    inside the backbone and have no per-frame vector to hand back.
    """
    import torch
    import torch.nn.functional as F

    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        if spec.family == "image_clip":
            inputs = processor(images=list(frames), return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            feats = _as_feature_tensor(model.get_image_features(**inputs))  # (n_frames, D)
            feats = F.normalize(feats.float(), dim=-1)
            per_frame = feats if return_frames else None
            vec = feats.mean(dim=0)
        elif spec.family == "xclip":
            per_frame = None
            inputs = _video_inputs(processor, frames)
            inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            vec = _as_feature_tensor(model.get_video_features(**inputs)).float().reshape(-1)
        elif spec.family == "videomae_v2":
            per_frame = None
            # This backbone is a Conv3d patch embed: it wants (B, C, T, H, W), not the
            # (B, T, C, H, W) every transformers video processor emits.
            inputs = _video_inputs(processor, frames)
            px = inputs["pixel_values"].to(device=device, dtype=dtype)
            if px.dim() != 5:
                raise RuntimeError(f"expected 5-D pixel_values for {spec.tag}, got {tuple(px.shape)}")
            px = px.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W) -> (B, C, T, H, W)
            vec = model.extract_features(px).float().reshape(-1)
        elif spec.family == "video_ssl":
            per_frame = None
            inputs = _video_inputs(processor, frames)
            inputs = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            out = model(**inputs)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out[0] if isinstance(out, (tuple, list)) else out
            hidden = hidden.float()
            if hidden.dim() == 3:  # (1, n_tokens, D) -> mean over tubelet tokens
                vec = hidden.mean(dim=1).reshape(-1)
            elif hidden.dim() == 2:  # (1, D)
                vec = hidden.reshape(-1)
            else:
                raise RuntimeError(f"unexpected hidden state shape {tuple(hidden.shape)} for {spec.tag}")
        else:
            raise ValueError(f"unknown family {spec.family!r}")
        vec = F.normalize(vec, dim=-1)
    out = vec.detach().cpu().numpy().astype(np.float32)
    if not return_frames:
        return out
    pf = None if per_frame is None else per_frame.detach().cpu().numpy().astype(np.float32)
    return out, pf


# ======================================================================================
# cache building
# ======================================================================================


def _shard_path(cache_dir: Path, cid: int) -> Path:
    return cache_dir / f"cond_{cid:03d}.npy"


def _frame_shard_path(cache_dir: Path, cid: int) -> Path:
    """Per-frame shard, kept beside the pooled one so both invalidate together."""
    return cache_dir / "frames" / f"cond_{cid:03d}.npy"


def build_cache(
    stim_root: Path,
    out_dir: Path,
    spec: ModelSpec,
    device: str = "cpu",
    n_frames: Optional[int] = None,
    force: bool = False,
    limit: Optional[int] = None,
    synthesize_flips: bool = False,
    fp16: bool = False,
    tag: Optional[str] = None,
    frame_order: str = "native",
    frame_order_seed: int = DEFAULT_FRAME_ORDER_SEED,
    verify_weights: bool = True,
    save_frame_emb: bool = False,
) -> Path:
    """Compute (or resume) the embedding cache and write ``{out_dir}/{tag}.npz``."""
    if frame_order not in FRAME_ORDERS:
        raise ValueError(f"frame_order must be one of {FRAME_ORDERS}, got {frame_order!r}")
    tag = tag or (spec.tag if frame_order == "native" else f"{spec.tag}__{frame_order}")
    n_frames = int(n_frames) if n_frames is not None else spec.n_frames
    # one row per condition, but keyed on the base video so a video's four
    # orientations share a scramble (see frame_permutation)
    perms = np.stack(
        [
            frame_permutation(frame_order, n_frames, frame_order_seed, cid // N_ORIENTATIONS)
            for cid in range(N_CONDITIONS)
        ]
    ).astype(np.int16)
    out_dir = Path(out_dir)
    cache_dir = out_dir / "_cache" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    stim_root = resolve_stimuli_root(Path(stim_root))
    if synthesize_flips:
        # only the original-orientation directory is required in this mode
        originals = [
            d
            for d in sorted(stim_root.iterdir())
            if d.is_dir() and "600ms" in d.name.lower() and "target" not in d.name.lower()
            and parse_orientation(d.name) == 0
        ]
        if len(originals) != 1:
            raise RuntimeError(f"expected exactly one original-orientation directory, got {originals}")
        orient_dirs = {0: originals[0]}
        index = index_videos(orient_dirs, strict=True)
    else:
        orient_dirs = discover_orientation_dirs(stim_root)
        index = index_videos(orient_dirs, strict=True)

    todo: List[int] = []
    for cid in range(N_CONDITIONS):
        if force or not _shard_path(cache_dir, cid).exists():
            todo.append(cid)
    if limit is not None:
        todo = todo[: int(limit)]
    print(
        f"[encode] model={tag} family={spec.family} n_frames={n_frames} device={device} "
        f"frame_order={frame_order}"
        + (f" seed={frame_order_seed}" if frame_order == "shuffled" else "")
        + f"\n[encode] stimuli={stim_root}\n"
        f"[encode] {N_CONDITIONS - len(todo)}/{N_CONDITIONS} conditions already cached, "
        f"{len(todo)} to compute"
    )

    n_decoded_by_cid: Dict[int, int] = {}
    weight_report: Dict[str, Any] = {"status": "not-loaded"}
    if todo:
        model, processor = load_model(spec, device, fp16=fp16, verify_weights=verify_weights)
        weight_report = getattr(model, "tactus_weight_report", {"status": "skipped"})
        t0 = time.time()
        # group by base video so a synthesized-flip run decodes each mp4 once
        by_video: Dict[int, List[int]] = {}
        for cid in todo:
            by_video.setdefault(cid // N_ORIENTATIONS + 1, []).append(cid)
        done = 0
        for vid in sorted(by_video):
            base_frames: Optional[List[np.ndarray]] = None
            for cid in sorted(by_video[vid]):
                orientation = cid % N_ORIENTATIONS
                if synthesize_flips:
                    if base_frames is None:
                        base_frames, n_dec = read_video_frames(index[condition_id(vid, 0)], n_frames)
                        n_decoded_by_cid[condition_id(vid, 0)] = n_dec
                    frames = flip_frames(base_frames, orientation)
                    n_dec = n_decoded_by_cid.get(condition_id(vid, 0), len(frames))
                else:
                    frames, n_dec = read_video_frames(index[cid], n_frames)
                n_decoded_by_cid[cid] = n_dec
                # frame-order control: within this clip only, after the resample, so the
                # permutation is exactly over the frames the encoder is about to see.
                if frame_order != "native":
                    frames = reorder_frames(frames, perms[cid])
                if save_frame_emb:
                    vec, pf = embed_clip(frames, spec, model, processor, device,
                                         return_frames=True)
                    if pf is not None:
                        fp = _frame_shard_path(cache_dir, cid)
                        fp.parent.mkdir(parents=True, exist_ok=True)
                        np.save(fp, pf)
                else:
                    vec = embed_clip(frames, spec, model, processor, device)
                np.save(_shard_path(cache_dir, cid), vec)
                done += 1
                if done % 20 == 0 or done == len(todo):
                    rate = done / max(time.time() - t0, 1e-6)
                    print(f"[encode]   {done}/{len(todo)} conditions  ({rate:.2f}/s)")

    # ---- assemble -------------------------------------------------------------------
    vecs: List[np.ndarray] = []
    missing: List[int] = []
    for cid in range(N_CONDITIONS):
        p = _shard_path(cache_dir, cid)
        if not p.exists():
            missing.append(cid)
            vecs.append(np.zeros(0, dtype=np.float32))
        else:
            vecs.append(np.load(p).astype(np.float32))
    if missing:
        raise RuntimeError(
            f"{len(missing)} conditions are still missing (e.g. {missing[:8]}); rerun without --limit"
        )
    dims = {v.shape[0] for v in vecs}
    if len(dims) != 1:
        raise RuntimeError(f"inconsistent embedding dims across conditions: {sorted(dims)}")
    cond_emb = np.stack(vecs).astype(np.float32)
    norms = np.linalg.norm(cond_emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(f"embeddings are not L2-normalized (norm range {norms.min()}..{norms.max()})")
    base_emb = cond_emb[[condition_id(v, 0) for v in range(1, N_VIDEOS + 1)]].copy()

    # Contract C extension (DECISIONS D18): the order-preserving array.  The
    # pooled cond_emb cannot answer any question about time -- its clip vector is
    # the arithmetic mean of these rows, so reversing or shuffling the frames
    # leaves it bit-identical.  frame_emb keeps the axis the mean deletes.
    frame_emb = None
    if save_frame_emb:
        fshards = [_frame_shard_path(cache_dir, cid) for cid in range(N_CONDITIONS)]
        if all(fp.exists() for fp in fshards):
            frame_emb = np.stack([np.load(fp).astype(np.float32) for fp in fshards])
            print(f"[encode]   frame_emb={frame_emb.shape}")
        else:
            n_missing = sum(1 for fp in fshards if not fp.exists())
            print(f"[encode]   frame_emb NOT written: {n_missing}/{N_CONDITIONS} frame "
                  f"shards missing. {spec.family!r} mixes frames inside the backbone "
                  "and has no per-frame vector; only the image_clip family does.")

    video_id = np.repeat(np.arange(1, N_VIDEOS + 1, dtype=np.int16), N_ORIENTATIONS)
    orientation = np.tile(np.arange(N_ORIENTATIONS, dtype=np.int8), N_VIDEOS)
    n_dec_arr = np.asarray(
        [n_decoded_by_cid.get(cid, -1) for cid in range(N_CONDITIONS)], dtype=np.int16
    )

    meta = {
        "model_tag": tag,
        "hf_id": spec.hf_id,
        "family": spec.family,
        "pooling": spec.pooling,
        "n_frames": n_frames,
        "embedding_dim": int(cond_emb.shape[1]),
        "has_frame_emb": bool(save_frame_emb),
        "normalized": "l2",
        "n_conditions": N_CONDITIONS,
        "n_videos": N_VIDEOS,
        "orientation_codes": {str(k): v for k, v in ORIENTATION_NAMES.items()},
        "condition_id_formula": "(video_id - 1) * 4 + orientation",
        "synthesized_flips": bool(synthesize_flips),
        "frame_order": frame_order,
        "frame_order_seed": int(frame_order_seed) if frame_order == "shuffled" else None,
        "frame_order_scope": "within-clip only; never mixes frames across clips and never "
                             "changes the video a condition_id maps to",
        "frame_perm_sha256": hashlib.sha256(perms.tobytes()).hexdigest(),
        "frame_perm_example_cid0": perms[0].tolist(),
        "weights": weight_report,
        "stimuli_root": str(stim_root),
        "orientation_dirs": {str(k): str(v) for k, v in sorted(orient_dirs.items())},
        "device": str(device),
        "fp16": bool(fp16),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": _version_info(),
        "notes": spec.notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{tag}.npz"
    np.savez_compressed(
        npz_path,
        cond_emb=cond_emb,
        base_emb=base_emb,
        meta=np.asarray(json.dumps(meta, indent=2)),
        video_id=video_id,
        orientation=orientation,
        n_frames_decoded=n_dec_arr,
        frame_perm=perms,
        **({"frame_emb": frame_emb} if frame_emb is not None else {}),
    )
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[encode] wrote {npz_path}\n"
        f"[encode]   cond_emb={cond_emb.shape}  base_emb={base_emb.shape}  dim={cond_emb.shape[1]}"
    )
    _report_sanity(cond_emb, base_emb)
    return npz_path


def _version_info() -> Dict[str, str]:
    info: Dict[str, str] = {"numpy": np.__version__}
    for mod in ("torch", "transformers", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = "unavailable"
    return info


def _report_sanity(cond_emb: np.ndarray, base_emb: np.ndarray) -> None:
    """Print the collapse check from the Phase-0 census, so a bad run is obvious at once."""
    sim = base_emb @ base_emb.T
    iu = np.triu_indices(base_emb.shape[0], k=1)
    off = sim[iu]
    centred = base_emb - base_emb.mean(0)
    sv = np.linalg.svd(centred, compute_uv=False)
    ev = sv**2 / max(float(np.sum(sv**2)), 1e-12)
    orient_sim = []
    for v in range(N_VIDEOS):
        block = cond_emb[v * N_ORIENTATIONS : (v + 1) * N_ORIENTATIONS]
        s = block @ block.T
        orient_sim.append(s[np.triu_indices(N_ORIENTATIONS, k=1)].mean())
    print(
        f"[encode]   pairwise cosine over 90 base videos: mean={off.mean():.3f} "
        f"min={off.min():.3f} max={off.max():.3f}\n"
        f"[encode]   PC variance: top3={ev[:3].sum():.3f} top10={ev[:10].sum():.3f}\n"
        f"[encode]   mean within-video across-orientation cosine: {np.mean(orient_sim):.3f}"
    )
    if off.mean() > 0.9 and ev[:3].sum() > 0.8:
        print("[encode]   WARNING: embedding collapse criterion met -- try a different encoder family")
    if np.mean(orient_sim) > 0.999:
        print(
            "[encode]   WARNING: orientations are near-identical in this embedding space; "
            "the orientation-equivariance analysis has nothing to work with here"
        )


def verify_flip_convention(stim_root: Path, n_check: int = 5) -> None:
    """Check that our synthesized flips match the shipped flipped files.

    ``horflip`` == left-right and ``vertflip`` == up-down is an assumption, not a
    documented fact of the dataset.  This decodes the first frame of a few videos in
    every orientation and reports the mean absolute difference against each candidate
    flip of the original: the correct convention should score near zero (exactly zero
    up to re-encoding noise) and the wrong one clearly higher.
    """
    stim_root = resolve_stimuli_root(Path(stim_root))
    orient_dirs = discover_orientation_dirs(stim_root)
    index = index_videos(orient_dirs, strict=True)
    print(f"[verify] comparing shipped flips against synthesized flips for {n_check} videos")
    for vid in range(1, int(n_check) + 1):
        orig, _ = read_video_frames(index[condition_id(vid, 0)], n_frames=1)
        base = orig[0].astype(np.float32)
        row = [f"video {vid:>2d}:"]
        for code in (1, 2, 3):
            shipped, _ = read_video_frames(index[condition_id(vid, code)], n_frames=1)
            ship = shipped[0].astype(np.float32)
            if ship.shape != base.shape:
                row.append(f"{ORIENTATION_NAMES[code]}=shape-mismatch")
                continue
            scores = {
                name: float(np.abs(ship - flip_frames([base], c)[0]).mean())
                for c, name in ((1, "LR"), (2, "UD"), (3, "LR+UD"))
            }
            best = min(scores, key=scores.get)
            row.append(f"{ORIENTATION_NAMES[code]}->best={best} ({scores[best]:.2f}) " + str(
                {k: round(v, 1) for k, v in scores.items()}
            ))
        print("[verify]   " + "  ".join(row))
    print(
        "[verify] expected: horflip->LR, vertflip->UD, horvertflip->LR+UD. "
        "If the mapping differs, fix flip_frames() and the orientation codes in the trial table."
    )


# ======================================================================================
# CLI
# ======================================================================================


def _list_models() -> None:
    print("Registered frozen video encoders:\n")
    for spec in MODEL_REGISTRY.values():
        print("  " + spec.describe().replace("\n", "\n  "))
        print()
    print("Any other checkpoint: --model-id <hf_id> --family {image_clip,xclip,video_ssl} "
          "--n-frames N [--tag NAME]\n")
    print(_UNSUPPORTED_NOTE)


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: D103
    ap = argparse.ArgumentParser(
        description="Build the frozen video-embedding cache (contract C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--list-models", action="store_true", help="print the encoder registry and exit")
    ap.add_argument("--stim-root", type=Path, help="dataset root, or the stimuli directory itself")
    ap.add_argument("--out", type=Path, default=Path("data/derived/video_emb"))
    ap.add_argument("--model", default="siglip2-base", help="registry tag")
    ap.add_argument("--model-id", default=None, help="arbitrary HF id (requires --family)")
    ap.add_argument(
        "--family", default=None, choices=["image_clip", "xclip", "video_ssl", "videomae_v2"]
    )
    ap.add_argument("--tag", default=None, help="output name; defaults to the model tag")
    ap.add_argument("--n-frames", type=int, default=None, help="override the model's frame count")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--fp16", action="store_true", help="half precision on CUDA")
    ap.add_argument("--force", action="store_true", help="recompute cached conditions")
    ap.add_argument("--save-frame-emb", action="store_true",
                    help="also write frame_emb (n_conditions, n_frames, D), the "
                         "order-preserving array of contract C (DECISIONS D18). The "
                         "pooled cond_emb is the arithmetic mean of these rows, so it "
                         "is bit-identical under any frame permutation and cannot "
                         "answer a question about time. Only the image_clip family "
                         "decomposes this way; for the video backbones the flag is a "
                         "no-op and the run says so.")
    ap.add_argument("--limit", type=int, default=None, help="compute at most N conditions (smoke test)")
    ap.add_argument(
        "--synthesize-flips",
        action="store_true",
        help="derive the 3 flipped orientations from the originals instead of decoding "
             "the shipped flipped files (only if those are unavailable)",
    )
    ap.add_argument(
        "--frame-order",
        default="native",
        choices=list(FRAME_ORDERS),
        help="temporal control applied WITHIN each clip after the resample to n_frames: "
             "'shuffled' = deterministic permutation drawn per base video (the 4 orientations "
             "of a video share it, so only time changes), 'reversed' = exact time reversal. "
             "Non-native orders default to the tag '{model}__{order}' so the native caches "
             "(contract C) are untouched; the permutation is stored in the npz as frame_perm.",
    )
    ap.add_argument(
        "--frame-order-seed",
        type=int,
        default=DEFAULT_FRAME_ORDER_SEED,
        help="seed for --frame-order shuffled (recorded in the meta)",
    )
    ap.add_argument(
        "--no-verify-weights",
        action="store_true",
        help="skip the checkpoint-vs-live-tensor audit (and the VideoMAE q_bias/v_bias repair)",
    )
    ap.add_argument("--dry-run", action="store_true", help="validate stimulus paths, load no model")
    ap.add_argument("--verify-flips", action="store_true", help="check the flip-axis convention")
    args = ap.parse_args(argv)

    if args.list_models:
        _list_models()
        return 0

    if args.stim_root is None:
        ap.error("--stim-root is required (or use --list-models)")

    if args.verify_flips:
        verify_flip_convention(args.stim_root)
        return 0

    if args.dry_run:
        root = resolve_stimuli_root(args.stim_root)
        dirs = discover_orientation_dirs(root)
        index = index_videos(dirs, strict=True)
        print(f"[dry-run] stimuli root: {root}")
        for code, d in sorted(dirs.items()):
            print(f"[dry-run]   {ORIENTATION_NAMES[code]:<12s} {len(list(d.glob('*.mp4'))):>3d} mp4  {d}")
        print(f"[dry-run] indexed {len(index)} / {N_CONDITIONS} conditions")
        sample = index[condition_id(1, 3)]
        print(f"[dry-run] example condition_id={condition_id(1, 3)} -> {sample.name}")
        return 0

    if args.model_id is not None:
        if args.family is None:
            ap.error("--model-id requires --family")
        spec = ModelSpec(
            tag=args.tag or re.sub(r"[^A-Za-z0-9]+", "_", args.model_id).strip("_"),
            hf_id=args.model_id,
            family=args.family,
            n_frames=args.n_frames if args.n_frames is not None else 15,
            pooling=f"{args.family} default pooling",
            trust_remote_code=True,
        )
    else:
        if args.model not in MODEL_REGISTRY:
            ap.error(f"unknown --model {args.model!r}; see --list-models")
        spec = MODEL_REGISTRY[args.model]

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    build_cache(
        stim_root=args.stim_root,
        out_dir=args.out,
        spec=spec,
        device=device,
        n_frames=args.n_frames,
        force=args.force,
        limit=args.limit,
        synthesize_flips=args.synthesize_flips,
        fp16=args.fp16,
        tag=args.tag,
        frame_order=args.frame_order,
        frame_order_seed=args.frame_order_seed,
        verify_weights=not args.no_verify_weights,
        save_frame_emb=args.save_frame_emb,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
