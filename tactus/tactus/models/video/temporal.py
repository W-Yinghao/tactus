#!/usr/bin/env python
"""Frame-**sequence** video cache (contract C-seq) -- the temporal sibling of contract C.

Contract C stores one pooled vector per condition, so a clip's temporal structure is
destroyed *before* the EEG model is ever shown it.  This module keeps the per-frame
axis:

``{out}/video_emb_seq/{tag}.npz``

* ``cond_seq``   ``(360, T, D)`` float32, indexed by ``condition_id = (video_id-1)*4 + orientation``
* ``base_seq``   ``(90, T, D)`` float32, indexed by ``video_id - 1`` (original orientation)
* ``cond_emb``   ``(360, D)`` float32 -- ``pool_reference(cond_seq)``, i.e. the *pooled*
  embedding **rederived from the sequence**.  Contract C to the bit, so a consumer can
  swap caches without knowing which one it holds, and so the sanity check below is a
  file-level comparison rather than a promise.
* ``base_emb``   ``(90, D)`` float32, same for the originals.
* ``meta``       JSON with ``frame_norm``, ``pool``, provenance, and the weight report.

**Contract C is not touched.**  This is a new directory and a new artifact.

The decomposition, per family
-----------------------------
The sequence is not a new definition of the embedding -- it is the *existing* pooled
embedding with the last reduction left undone, so that

    ``l2(mean_t seq[t]) == cond_emb_of_contract_C``

holds by construction and is checked numerically by ``--verify``:

``image_clip`` (SigLIP2, SigLIP, CLIP)
    ``encode.embed_clip`` computes ``per-frame get_image_features -> L2 -> mean -> L2``.
    We stop after the per-frame L2.  ``frame_norm = "l2"``; every row of ``seq`` is a unit
    vector, ``pool = "mean+l2"``.  Exact.
``video_ssl`` (VideoMAE v1, V-JEPA 2, InternVideo2)
    ``encode.embed_clip`` computes ``mean over ALL tokens of last_hidden_state -> L2``.
    The token axis of a tubelet ViT is time-major -- ``(T', S, D)`` with ``T'`` temporal
    positions and ``S`` spatial patches each -- so the mean over all tokens is exactly the
    mean over ``T'`` of the per-time spatial means.  We stop after the spatial mean.
    ``frame_norm = "none"`` (normalising per time step would *not* reproduce the pooled
    vector), ``pool = "mean+l2"``.  Exact.  Note ``T'`` is the number of *tubelets*
    (VideoMAE: ``n_frames / 2``), not frames -- recorded as ``n_steps`` in the meta.
``xclip``
    Refused.  X-CLIP's ``get_video_features`` runs a multi-frame integration transformer
    and a prompt-conditioned projection; no per-frame quantity averages back to it, so any
    "sequence" we invented here would silently be a *different* representation and the
    sanity check would be a lie.  Measure X-CLIP through contract C.

Frame order is a *within-clip* manipulation of exactly the frames the encoder sees; the
shuffling used by the analysis below is applied to the stored sequence (no re-encode
needed -- that is the entire point of keeping the axis).

CLI
---
::

    # build (CPU; resumable; originals first so the 90-video analyses can start early)
    python -m tactus.models.video.temporal build --stim-root $TACTUS_BIDS_ROOT \\
        --out $TACTUS_DATA_ROOT/derived/video_emb_seq --model siglip2-base --device cpu

    # the sanity check: does mean-pooling the sequence reproduce contract C?
    python -m tactus.models.video.temporal verify \\
        --seq $TACTUS_DATA_ROOT/derived/video_emb_seq/siglip2-base.npz \\
        --pooled $TACTUS_DATA_ROOT/derived/video_emb/siglip2-base.npz

    # does frame order carry anything? (RDM native-vs-shuffled + LOVO pooled-vs-temporal)
    python -m tactus.models.video.temporal analyze \\
        --seq $TACTUS_DATA_ROOT/derived/video_emb_seq/siglip2-base.npz \\
        --vtd $TACTUS_BIDS_ROOT/code/analysis/VTD.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encode import (
    DEFAULT_FRAME_ORDER_SEED,
    MODEL_REGISTRY,
    N_CONDITIONS,
    N_ORIENTATIONS,
    N_VIDEOS,
    ModelSpec,
    _as_feature_tensor,
    _video_inputs,
    condition_id,
    discover_orientation_dirs,
    flip_frames,
    frame_permutation,
    index_videos,
    load_model,
    read_video_frames,
    resolve_stimuli_root,
)

__all__ = [
    "SEQ_FAMILIES",
    "frame_sequence",
    "pool_reference",
    "build_sequence_cache",
    "load_sequence_cache",
    "verify_against_pooled",
    "shuffle_sequence",
    "temporal_readouts",
]

#: Families whose pooled embedding decomposes exactly into a per-step sequence.
SEQ_FAMILIES: Tuple[str, ...] = ("image_clip", "video_ssl")

#: ``frame_norm`` per family: what one row of ``seq`` is.
_FRAME_NORM: Dict[str, str] = {"image_clip": "l2", "video_ssl": "none"}


# ======================================================================================
# encoding: the pooled embedding with the last reduction left undone
# ======================================================================================


def frame_sequence(
    frames: Sequence[np.ndarray], spec: ModelSpec, model, processor, device: str
) -> np.ndarray:
    """Embed one clip to ``(T, D)`` float32 -- the un-pooled form of ``encode.embed_clip``.

    See the module docstring for the per-family decomposition.  ``T`` is the number of
    frames for ``image_clip`` and the number of *tubelets* for ``video_ssl``.
    """
    import torch
    import torch.nn.functional as F

    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        if spec.family == "image_clip":
            inputs = processor(images=list(frames), return_tensors="pt")
            inputs = {
                k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()
            }
            feats = _as_feature_tensor(model.get_image_features(**inputs))  # (T, D)
            seq = F.normalize(feats.float(), dim=-1)
        elif spec.family == "video_ssl":
            inputs = _video_inputs(processor, frames)
            inputs = {
                k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()
            }
            out = model(**inputs)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out[0] if isinstance(out, (tuple, list)) else out
            hidden = hidden.float()
            if hidden.dim() != 3:
                raise RuntimeError(
                    f"{spec.tag}: expected (1, n_tokens, D) last_hidden_state for a sequence "
                    f"read-out, got {tuple(hidden.shape)}"
                )
            n_steps = _temporal_positions(spec, model, hidden.shape[1])
            n_tok = hidden.shape[1]
            if n_tok % n_steps:
                raise RuntimeError(
                    f"{spec.tag}: {n_tok} tokens do not divide into {n_steps} temporal "
                    "positions; the token layout is not time-major and the sequence would "
                    "not pool back to contract C"
                )
            # (1, T'*S, D) -> (T', S, D) -> mean over the spatial patches of each tubelet
            seq = hidden.reshape(n_steps, n_tok // n_steps, hidden.shape[-1]).mean(dim=1)
        else:
            raise NotImplementedError(
                f"family {spec.family!r} has no exact per-step decomposition of its pooled "
                f"embedding (see the module docstring); supported: {SEQ_FAMILIES}"
            )
    return seq.detach().cpu().numpy().astype(np.float32)


def _temporal_positions(spec: ModelSpec, model, n_tokens: int) -> int:
    """Number of temporal positions in a tubelet ViT's token axis."""
    cfg = getattr(model, "config", None)
    tubelet = int(getattr(cfg, "tubelet_size", 0) or 0)
    n_frames = int(getattr(cfg, "num_frames", 0) or 0) or int(spec.n_frames)
    if tubelet > 0:
        return max(n_frames // tubelet, 1)
    # V-JEPA-style configs without tubelet_size: fall back to the largest divisor of
    # n_tokens that does not exceed n_frames.
    for t in range(min(n_frames, n_tokens), 0, -1):
        if n_tokens % t == 0:
            return t
    return 1


def pool_reference(seq: np.ndarray) -> np.ndarray:
    """``(..., T, D) -> (..., D)``: mean over the step axis, then L2.

    This is the *definition* of the contract-C pooled embedding in terms of the
    sequence, for every family in :data:`SEQ_FAMILIES`.  Both families end in
    ``mean -> L2``; they differ only in what one step is (see the module docstring).
    """
    v = np.asarray(seq, dtype=np.float64).mean(axis=-2)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, 1e-12)).astype(np.float32)


# ======================================================================================
# cache building
# ======================================================================================


def _shard(cache_dir: Path, cid: int) -> Path:
    return cache_dir / f"cond_{cid:03d}.npy"


def _condition_order() -> List[int]:
    """Originals (orientation 0) first, then the flips.

    The 90x90 RDM and the LOVO census metric only need the originals, so this ordering
    makes the analysis runnable after 25% of a long CPU job instead of 100%.
    """
    firsts = [condition_id(v, 0) for v in range(1, N_VIDEOS + 1)]
    rest = [c for c in range(N_CONDITIONS) if c % N_ORIENTATIONS != 0]
    return firsts + rest


def build_sequence_cache(
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
    verify_weights: bool = True,
    partial_ok: bool = False,
) -> Path:
    """Compute (or resume) the frame-sequence cache and write ``{out_dir}/{tag}.npz``.

    ``partial_ok`` assembles an npz from whatever shards exist, zero-filling the rest and
    recording ``complete: false`` plus the list of missing condition ids in the meta.  It
    exists so the 90 originals can be analysed while the 270 flips are still encoding; a
    partial file must never be fed to training.
    """
    if spec.family not in SEQ_FAMILIES:
        raise NotImplementedError(
            f"family {spec.family!r} is not sequence-decomposable; supported: {SEQ_FAMILIES}"
        )
    tag = tag or spec.tag
    n_frames = int(n_frames) if n_frames is not None else spec.n_frames
    out_dir = Path(out_dir)
    cache_dir = out_dir / "_cache" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    stim_root = resolve_stimuli_root(Path(stim_root))
    if synthesize_flips:
        originals = [
            d
            for d in sorted(stim_root.iterdir())
            if d.is_dir()
            and "600ms" in d.name.lower()
            and "target" not in d.name.lower()
            and "flip" not in d.name.lower()
        ]
        if len(originals) != 1:
            raise RuntimeError(f"expected exactly one original-orientation directory, got {originals}")
        orient_dirs = {0: originals[0]}
    else:
        orient_dirs = discover_orientation_dirs(stim_root)
    index = index_videos(orient_dirs, strict=True)

    todo = [c for c in _condition_order() if force or not _shard(cache_dir, c).exists()]
    n_present = N_CONDITIONS - len(todo)
    if limit is not None:
        todo = todo[: int(limit)]
    print(
        f"[seq] model={tag} family={spec.family} n_frames={n_frames} device={device}\n"
        f"[seq] stimuli={stim_root}\n"
        f"[seq] {n_present}/{N_CONDITIONS} conditions already cached, "
        f"{len(todo)} to compute",
        flush=True,
    )

    n_decoded: Dict[int, int] = {}
    weight_report: Dict[str, Any] = {"status": "not-loaded"}
    if todo:
        model, processor = load_model(spec, device, fp16=fp16, verify_weights=verify_weights)
        weight_report = getattr(model, "tactus_weight_report", {"status": "skipped"})
        t0 = time.time()
        base_cache: Dict[int, List[np.ndarray]] = {}
        for done, cid in enumerate(todo, start=1):
            vid, orientation = cid // N_ORIENTATIONS + 1, cid % N_ORIENTATIONS
            if synthesize_flips:
                if vid not in base_cache:
                    base_cache.clear()
                    fr, nd = read_video_frames(index[condition_id(vid, 0)], n_frames)
                    base_cache[vid] = fr
                    n_decoded[condition_id(vid, 0)] = nd
                frames = flip_frames(base_cache[vid], orientation)
                nd = n_decoded.get(condition_id(vid, 0), len(frames))
            else:
                frames, nd = read_video_frames(index[cid], n_frames)
            n_decoded[cid] = nd
            seq = frame_sequence(frames, spec, model, processor, device)
            np.save(_shard(cache_dir, cid), seq)
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - done) / max(rate, 1e-9) / 60.0
                print(f"[seq]   {done}/{len(todo)}  ({rate:.3f}/s, ETA {eta:.1f} min)", flush=True)

    # ---- assemble --------------------------------------------------------------------
    shards: Dict[int, np.ndarray] = {}
    missing: List[int] = []
    for cid in range(N_CONDITIONS):
        p = _shard(cache_dir, cid)
        if p.exists():
            shards[cid] = np.load(p).astype(np.float32)
        else:
            missing.append(cid)
    if missing and not partial_ok:
        raise RuntimeError(
            f"{len(missing)} conditions are still missing (e.g. {missing[:8]}); "
            "rerun without --limit, or pass --partial-ok for an analysis-only artifact"
        )
    if not shards:
        raise RuntimeError("no shards on disk; nothing to assemble")
    shapes = {v.shape for v in shards.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"inconsistent sequence shapes across conditions: {sorted(shapes)}")
    n_steps, dim = next(iter(shapes))

    cond_seq = np.zeros((N_CONDITIONS, n_steps, dim), dtype=np.float32)
    for cid, v in shards.items():
        cond_seq[cid] = v
    base_seq = cond_seq[[condition_id(v, 0) for v in range(1, N_VIDEOS + 1)]].copy()
    cond_emb = pool_reference(cond_seq)
    base_emb = cond_emb[[condition_id(v, 0) for v in range(1, N_VIDEOS + 1)]].copy()

    frame_norm = _FRAME_NORM[spec.family]
    if frame_norm == "l2":
        fn = np.linalg.norm(cond_seq[sorted(shards)], axis=-1)
        if not np.allclose(fn, 1.0, atol=1e-3):
            raise RuntimeError(f"per-frame vectors are not unit norm ({fn.min()}..{fn.max()})")

    meta = {
        "artifact": "video_emb_seq",
        "model_tag": tag,
        "hf_id": spec.hf_id,
        "family": spec.family,
        "pooling_of_contract_C": spec.pooling,
        "sequence_decomposition": (
            "per-frame get_image_features -> L2  (pool: mean -> L2)"
            if spec.family == "image_clip"
            else "per-tubelet spatial mean of last_hidden_state  (pool: mean -> L2)"
        ),
        "frame_norm": frame_norm,
        "pool": "mean+l2",
        "n_frames": n_frames,
        "n_steps": int(n_steps),
        "steps_are": "frames" if spec.family == "image_clip" else "tubelets",
        "embedding_dim": int(dim),
        "n_conditions": N_CONDITIONS,
        "n_videos": N_VIDEOS,
        "condition_id_formula": "(video_id - 1) * 4 + orientation",
        "complete": not missing,
        "n_missing": len(missing),
        "missing_condition_ids": missing[:512],
        "synthesized_flips": bool(synthesize_flips),
        "weights": weight_report,
        "stimuli_root": str(stim_root),
        "device": str(device),
        "fp16": bool(fp16),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": _versions(),
        "note": "cond_emb/base_emb here are pool_reference(cond_seq) and must match "
                "contract C; see `temporal.py verify`.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{tag}.npz"
    np.savez_compressed(
        npz_path,
        cond_seq=cond_seq,
        base_seq=base_seq,
        cond_emb=cond_emb,
        base_emb=base_emb,
        meta=np.asarray(json.dumps(meta, indent=2)),
        video_id=np.repeat(np.arange(1, N_VIDEOS + 1, dtype=np.int16), N_ORIENTATIONS),
        orientation=np.tile(np.arange(N_ORIENTATIONS, dtype=np.int8), N_VIDEOS),
        n_frames_decoded=np.asarray(
            [n_decoded.get(c, -1) for c in range(N_CONDITIONS)], dtype=np.int16
        ),
        cached=np.asarray([c in shards for c in range(N_CONDITIONS)], dtype=bool),
    )
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[seq] wrote {npz_path}\n"
        f"[seq]   cond_seq={cond_seq.shape} base_seq={base_seq.shape} "
        f"complete={not missing} ({len(shards)}/{N_CONDITIONS} cached)",
        flush=True,
    )
    return npz_path


def _versions() -> Dict[str, str]:
    info: Dict[str, str] = {"numpy": np.__version__}
    for mod in ("torch", "transformers", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = "unavailable"
    return info


def load_sequence_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """``(cond_seq, base_seq, meta)`` from a ``video_emb_seq`` npz."""
    z = np.load(Path(path), allow_pickle=True)
    meta: Dict[str, Any] = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:  # noqa: BLE001
            meta = {}
    return np.asarray(z["cond_seq"], np.float32), np.asarray(z["base_seq"], np.float32), meta


# ======================================================================================
# the sanity check
# ======================================================================================


def verify_against_pooled(seq_npz: Path, pooled_npz: Path) -> Dict[str, Any]:
    """Does ``mean-pool(sequence)`` reproduce contract C?  Numbers, not promises.

    Reports max |diff|, min cosine and the norm range over the conditions actually
    cached.  A cache built on a different device/dtype than contract C will not agree to
    float32 epsilon -- fp16 on CUDA costs ~1e-6 of cosine -- so the report carries both
    caches' ``device``/``fp16`` fields and the caller judges.
    """
    zs = np.load(Path(seq_npz), allow_pickle=True)
    zp = np.load(Path(pooled_npz), allow_pickle=True)
    seq_meta = json.loads(str(zs["meta"])) if "meta" in zs else {}
    pool_meta = json.loads(str(zp["meta"])) if "meta" in zp else {}

    cond_seq = np.asarray(zs["cond_seq"], np.float64)
    cached = (
        np.asarray(zs["cached"], bool)
        if "cached" in zs
        else np.ones(len(cond_seq), dtype=bool)
    )
    ref = np.asarray(zp["cond_emb"], np.float64)
    if ref.shape[0] != cond_seq.shape[0]:
        raise ValueError(f"condition count mismatch: {ref.shape[0]} vs {cond_seq.shape[0]}")
    if ref.shape[1] != cond_seq.shape[2]:
        raise ValueError(f"dim mismatch: pooled D={ref.shape[1]} vs sequence D={cond_seq.shape[2]}")

    got = pool_reference(cond_seq[cached]).astype(np.float64)
    exp = ref[cached]
    diff = np.abs(got - exp)
    cos = (got * exp).sum(1) / (
        np.linalg.norm(got, axis=1) * np.linalg.norm(exp, axis=1) + 1e-12
    )
    same_device = (
        str(seq_meta.get("device")) == str(pool_meta.get("device"))
        and bool(seq_meta.get("fp16")) == bool(pool_meta.get("fp16"))
    )
    return {
        "n_compared": int(cached.sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "max_angle_deg": float(np.degrees(np.arccos(np.clip(cos.min(), -1, 1)))),
        "seq_device": f"{seq_meta.get('device')} fp16={seq_meta.get('fp16')}",
        "pooled_device": f"{pool_meta.get('device')} fp16={pool_meta.get('fp16')}",
        "same_numerics": bool(same_device),
        "verdict": (
            "EXACT (same numerics)"
            if same_device and diff.max() < 1e-5
            else "AGREES up to device/dtype"
            if cos.min() > 1 - 1e-4
            else "DISAGREES -- more than rounding separates these two caches. Either the "
                 "sequence path preprocesses differently, or the two caches were produced "
                 "by different *weights* (e.g. one before and one after the VideoMAE "
                 "q_bias/v_bias repair). Settle it by running both paths in one process on "
                 "one model object: that comparison isolates the path from the weights."
        ),
    }


# ======================================================================================
# frame-order manipulations on the stored sequence
# ======================================================================================


def shuffle_sequence(
    seq: np.ndarray, order: str = "shuffled", seed: int = DEFAULT_FRAME_ORDER_SEED
) -> np.ndarray:
    """Reorder the step axis of ``(N, T, D)`` per item, using ``encode.frame_permutation``.

    Item ``i`` is keyed by ``i`` itself, so a base-video-indexed array (90) gets one draw
    per video and a condition-indexed array (360) gets one draw per condition; pass
    ``base_seq`` when the four orientations of a video must share a scramble.
    """
    seq = np.asarray(seq)
    if seq.ndim != 3:
        raise ValueError(f"expected (N, T, D), got {seq.shape}")
    out = np.empty_like(seq)
    for i in range(seq.shape[0]):
        out[i] = seq[i][frame_permutation(order, seq.shape[1], seed, i)]
    return out


def temporal_readouts(
    seq: np.ndarray, n_random: int = 8, seed: int = 0
) -> Dict[str, np.ndarray]:
    """Parameter-free read-outs of ``(N, T, D)``, each returned L2-normalised ``(N, D')``.

    Deliberately *not* learned.  A randomly-initialised attention head mixes "is there
    order-dependent variance" with "did this seed happen to find it"; these are
    closed-form, so each number is a property of the encoder's output.

    The set is chosen to be *complete*, not illustrative.  Every linear read-out of a
    frame sequence is ``z = sum_t a_t seq[t]`` for some ``a in R^T``; the constant
    direction of that space is the mean and the orthogonal complement is everything
    order-sensitive.  So:

    ``mean``
        The order-blind control (``a = 1/T``).  Equals contract C exactly, so any
        read-out whose RDM matches this one's is temporal in name only.
    ``slope``, ``delta``
        Two named order-sensitive contrasts: a centred linear ramp (the clip's temporal
        trend) and ``seq[-1] - seq[0]`` (net displacement).  Time reversal flips both.
    ``randlin``
        A random direction ``a ~ N(0, I_T)``: a *typical* linear read-out, averaged over
        ``n_random`` draws by the caller.  Scale-free, so unlike a random network it
        cannot under-report by being initialised small.
    ``resflat``
        ``concat_t(seq[t] - mean)``, the whole order-sensitive part of the clip with the
        pooled component projected out.  Nothing order-sensitive is discarded, so this is
        the **ceiling for the temporal-only information** at this metric.
    ``flat``
        ``concat_t(seq[t])``, the full sequence.  Every read-out -- linear, attention,
        conv, transformer -- is a function of this, so its score is the **ceiling for the
        whole family**.  If ``flat`` does not beat ``mean``, no read-out can.
    ``meanslope``
        ``concat(mean, slope)``: what a linear head allowed to use time would see with the
        pooled information still present.
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 3:
        raise ValueError(f"expected (N, T, D), got {seq.shape}")
    N, T, D = seq.shape

    def n(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

    mean = seq.mean(1)
    w = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    w /= np.linalg.norm(w) + 1e-12
    slope = np.einsum("t,ntd->nd", w, seq)
    resid = seq - seq.mean(1, keepdims=True)

    out = {
        "mean": n(mean),
        "slope": n(slope),
        "delta": n(seq[:, -1] - seq[:, 0]),
        "meanslope": n(np.hstack([n(mean), n(slope)])),
        "resflat": n(resid.reshape(N, T * D)),
        "flat": n(seq.reshape(N, T * D)),
    }
    rng = np.random.default_rng(seed)
    for j in range(int(n_random)):
        a = rng.standard_normal(T)
        a -= a.mean()  # orthogonal to the mean: a *purely* order-sensitive draw
        out[f"randlin{j}"] = n(np.einsum("t,ntd->nd", a, seq))
    return out


def learned_readouts(
    seq: np.ndarray, n_seeds: int = 8, seed: int = 0, out_dim: Optional[int] = None
) -> Dict[str, List[np.ndarray]]:
    """Randomly-initialised temporal heads from :mod:`tactus.models.heads`, one list per pool.

    ``init_as_mean`` is turned **off** here on purpose: the point of the analysis is to ask
    whether a temporal read-out *can* see something the mean cannot, so the head must not
    start out equal to the mean.  ``out_dim = D_vid`` keeps the projector square, and the
    projector is shared across seeds' arms only in shape -- each seed redraws everything.

    Random weights answer "is there order-dependent variance at all", never "is it useful";
    the LOVO and RSA sections answer the second question.
    """
    import torch

    from ...models.heads import VideoSequenceProjector

    n, T, D = seq.shape
    out_dim = int(out_dim or D)
    x = torch.from_numpy(np.ascontiguousarray(seq, dtype=np.float32))
    out: Dict[str, List[np.ndarray]] = {}
    for pool in ("posw", "attn", "tconv", "set"):
        arms: List[np.ndarray] = []
        for s in range(int(n_seeds)):
            torch.manual_seed(int(seed) * 1000 + s)
            head = VideoSequenceProjector(
                in_dim=D,
                out_dim=out_dim,
                n_frames=T,
                pool="attn" if pool == "set" else pool,
                learn_pos=pool != "set",
                init_as_mean=False,
            ).eval()
            with torch.no_grad():
                arms.append(head(x).numpy().astype(np.float64))
        out[pool] = arms
    return out


def _rdm(emb: np.ndarray) -> np.ndarray:
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return 1.0 - e @ e.T


def _vec(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats

    return float(stats.spearmanr(a, b).statistic)


def _order_effect(emb_a: np.ndarray, emb_b: np.ndarray) -> Dict[str, float]:
    """How far apart are two embeddings of the same 90 videos, as vectors and as geometry."""
    a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12)
    b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12)
    cos = (a * b).sum(1)
    ra, rb = _vec(_rdm(a)), _vec(_rdm(b))
    return {
        "cos_native_vs_shuffled_mean": float(cos.mean()),
        "cos_native_vs_shuffled_min": float(cos.min()),
        "rdm_spearman": _spearman(ra, rb),
        "rdm_mean_abs_delta": float(np.abs(ra - rb).mean()),
        "rdm_scale": float(ra.mean()),
    }


def _rowspace(emb: np.ndarray) -> np.ndarray:
    """Exact isometric re-coordinatisation of ``N`` vectors into ``<= N`` dimensions.

    ``US`` from the thin SVD satisfies ``(US)(US)^T == emb emb^T`` exactly, so every
    inner product, cosine, norm and RDM among the ``N`` rows is preserved to numerical
    precision -- and so is any ridge prediction, because ridge is equivariant under an
    orthogonal change of basis of the target.  ``flat`` is 90x11520 but only spans 90
    dimensions; carrying the other 11430 is pure cost, and dropping them lets every
    read-out have its *own* permutation null instead of borrowing the pooled one.
    """
    e = np.asarray(emb, dtype=np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    if e.shape[1] <= e.shape[0]:
        return e
    u, s, _ = np.linalg.svd(e, full_matrices=False)
    return u * s


def _lovo(emb: np.ndarray, X: np.ndarray, alpha: float = 10.0) -> Dict[str, float]:
    """The census metric: ridge from VTD attributes to a held-out embedding, cosine retrieval."""
    Y = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    n = len(Y)
    ranks = np.empty(n, dtype=int)
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, Ytr = X[tr], Y[tr]
        xm, ym = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
        Xc, Yc = Xtr - xm, Ytr - ym
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ Yc)
        pred = (X[i : i + 1] - xm) @ W + ym
        pred = pred / (np.linalg.norm(pred) + 1e-12)
        sims = (Y @ pred.T).ravel()
        ranks[i] = int(np.sum(sims > sims[i]))
    return {
        "top1": float(np.mean(ranks == 0)),
        "top5": float(np.mean(ranks < 5)),
        "mean_rank": float(ranks.mean() + 1),
    }


def _lovo_null(emb: np.ndarray, X: np.ndarray, n_perm: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    t1 = np.empty(n_perm)
    t5 = np.empty(n_perm)
    for j in range(int(n_perm)):
        r = _lovo(emb, X[rng.permutation(len(X))])
        t1[j], t5[j] = r["top1"], r["top5"]
    return {
        "top1_null_mean": float(t1.mean()),
        "top5_null_mean": float(t5.mean()),
        "top5_null_p95": float(np.quantile(t5, 0.95)),
        "_top1_null": t1,
        "_top5_null": t5,
    }


def _directional_probe(emb: np.ndarray, y: np.ndarray, seed: int = 0, n_perm: int = 2000) -> Dict[str, float]:
    """Leave-one-video-out AUC for a binary attribute from the embedding, ridge-regressed.

    ``approaching`` (yes/no) is the one VTD attribute that *is* a statement about the
    direction of time: an approaching hand played backwards is a retreating hand.  If a
    temporal read-out carries anything the pooled one does not, this is where it shows.
    """
    from scipy import stats

    Y = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    n = len(Y)
    yy = np.asarray(y, dtype=float)
    score = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, ytr = Y[tr], yy[tr]
        xm, ym = Xtr.mean(0, keepdims=True), ytr.mean()
        Xc = Xtr - xm
        # dual (kernel) ridge: the embedding can be 11520-dimensional (`flat`), so solve
        # in sample space (89 x 89) rather than feature space.  Identical solution.
        K = Xc @ Xc.T
        a = np.linalg.solve(K + 10.0 * np.eye(len(K)), ytr - ym)
        score[i] = float((Y[i] - xm.ravel()) @ (Xc.T @ a) + ym)
    pos, neg = score[yy > 0.5], score[yy <= 0.5]
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(u.statistic / (len(pos) * len(neg)))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for j in range(int(n_perm)):
        yp = rng.permutation(yy)
        null[j] = stats.mannwhitneyu(score[yp > 0.5], score[yp <= 0.5]).statistic / (
            int((yp > 0.5).sum()) * int((yp <= 0.5).sum())
        )
    return {
        "auc": auc,
        "p_label_perm": float((1 + int(np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)))) / (n_perm + 1)),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }


def _collapse_random(rows: Dict[str, Dict[str, Any]], n_random: int) -> Dict[str, Dict[str, Any]]:
    """Fold the ``randlin{j}`` arms into a single ``randlin`` row (mean, with ``_sd``)."""
    keys = [f"randlin{j}" for j in range(int(n_random)) if f"randlin{j}" in rows]
    if not keys:
        return rows
    arms = [rows.pop(k) for k in keys]
    agg: Dict[str, Any] = {}
    for field in arms[0]:
        vals = [a[field] for a in arms]
        if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            agg[field] = float(np.mean(vals))
            agg[field + "_sd"] = float(np.std(vals))
        else:
            agg[field] = vals[0]
    agg["kind"] = f"closed-form, {len(keys)} random draws"
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in rows.items():
        out[k] = v
        if k == "delta":  # keep the random row next to the other order-sensitive ones
            out["randlin"] = agg
    if "randlin" not in out:
        out["randlin"] = agg
    return out


def gallery_separability(emb: np.ndarray) -> Dict[str, float]:
    """How distinguishable are the 90 targets from each other under this read-out?

    The D4 primary endpoint is retrieval against an 18-way source-video gallery, so the
    *target geometry* is half the problem: a gallery whose members sit at cosine 0.88 from
    each other asks the EEG for a much finer discrimination than one at 0.03.  This is a
    property of the video embedding alone -- no EEG, no training -- and it is the reason a
    read-out can matter even when it adds nothing the behavioural attributes can name.
    """
    e = np.asarray(emb, dtype=np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    g = e @ e.T
    off = ~np.eye(len(e), dtype=bool)
    v = g[off]
    nn = g.copy()
    np.fill_diagonal(nn, -np.inf)
    centred = e - e.mean(0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False) ** 2
    sv = sv / max(sv.sum(), 1e-12)
    return {
        "mean_off_diagonal_cosine": float(v.mean()),
        "max_off_diagonal_cosine": float(v.max()),
        "nearest_neighbour_cosine_mean": float(nn.max(1).mean()),
        "effective_rank": float(np.exp(-(sv * np.log(sv + 1e-12)).sum())),
    }


def orientation_reliability(cond_seq: np.ndarray, cached: np.ndarray) -> Dict[str, Any]:
    """Is the temporal component signal, or is it encoder noise?

    The four orientations of a video are the *same clip*, mirrored.  Mirroring is a purely
    spatial operation, so the clip's dynamics -- how much and how fast the image changes
    from frame to frame -- are identical across all four.  Anything in the temporal axis
    that is a property of the *event* must therefore repeat across orientations; anything
    that does not repeat is the encoder reacting to pixel detail.

    Two probes, because they fail differently:

    ``residual cosine``
        ``cos`` between the flattened order-sensitive residual ``seq[t] - mean`` of
        orientation 0 and of orientation ``o``, against an across-*video* baseline.
        A mirror does change what SigLIP2 sees, so a *low* value here is ambiguous; a
        *high* value is conclusive evidence of reproducible temporal structure.
    ``motion profile``
        Pearson ``r`` between the frame-to-frame displacement curves
        ``||seq[t+1] - seq[t]||``.  This is a scalar time series of how fast the clip is
        changing, and mirroring should leave it essentially untouched -- so here a low
        value *is* evidence that the temporal axis is not tracking the event.
    """
    cached = np.asarray(cached, dtype=bool)
    have = [
        v
        for v in range(N_VIDEOS)
        if all(cached[v * N_ORIENTATIONS + o] for o in range(N_ORIENTATIONS))
    ]
    if len(have) < 4:
        return {"n_videos_with_all_orientations": len(have), "note": "too few to measure"}
    seq = np.asarray(cond_seq, dtype=np.float64)
    T = seq.shape[1]

    def unitflat(x: np.ndarray) -> np.ndarray:
        r = x - x.mean(-2, keepdims=True)
        r = r.reshape(*r.shape[:-2], -1)
        return r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-12)

    def motion(x: np.ndarray) -> np.ndarray:
        m = np.linalg.norm(np.diff(x, axis=-2), axis=-1)  # (..., T-1)
        m = m - m.mean(-1, keepdims=True)
        return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)

    ids = np.asarray(have)
    o0 = seq[ids * N_ORIENTATIONS]
    r0, m0 = unitflat(o0), motion(o0)
    within_r: List[float] = []
    within_m: List[float] = []
    for o in range(1, N_ORIENTATIONS):
        ov = seq[ids * N_ORIENTATIONS + o]
        within_r.append(float((r0 * unitflat(ov)).sum(-1).mean()))
        within_m.append(float((m0 * motion(ov)).sum(-1).mean()))
    off = ~np.eye(len(ids), dtype=bool)
    across_r = float((r0 @ r0.T)[off].mean())
    across_m = float((m0 @ m0.T)[off].mean())
    # the same two numbers for the pooled component, as the reference scale
    p0 = o0.mean(-2)
    p0 = p0 / (np.linalg.norm(p0, axis=-1, keepdims=True) + 1e-12)
    pooled_within = []
    for o in range(1, N_ORIENTATIONS):
        pv = seq[ids * N_ORIENTATIONS + o].mean(-2)
        pv = pv / (np.linalg.norm(pv, axis=-1, keepdims=True) + 1e-12)
        pooled_within.append(float((p0 * pv).sum(-1).mean()))
    return {
        "n_videos_with_all_orientations": len(have),
        "n_steps": int(T),
        "residual_cos_same_video_across_orientation": float(np.mean(within_r)),
        "residual_cos_across_video": across_r,
        "motion_profile_r_same_video_across_orientation": float(np.mean(within_m)),
        "motion_profile_r_across_video": across_m,
        "pooled_cos_same_video_across_orientation": float(np.mean(pooled_within)),
        "pooled_cos_across_video": float((p0 @ p0.T)[off].mean()),
        "per_orientation_residual_cos": [round(v, 4) for v in within_r],
        "per_orientation_motion_r": [round(v, 4) for v in within_m],
    }


def run_analysis(
    seq_npz: Path,
    vtd_path: Path,
    seed: int = 0,
    n_seeds: int = 8,
    n_perm: int = 200,
) -> Dict[str, Any]:
    """Does frame order carry anything?  RDM native-vs-shuffled + the census LOVO metric.

    Everything runs on the 90 **original-orientation** videos, which is the unit the
    census and the primary endpoint's gallery use, and the unit the permutation nulls
    treat as exchangeable.
    """
    import pandas as pd

    from ...data.events import load_vtd
    from ...eval.census import _design_matrix

    cond_seq, base_seq, meta = load_sequence_cache(seq_npz)
    z = np.load(Path(seq_npz), allow_pickle=True)
    cached_flags = np.asarray(z["cached"], bool) if "cached" in z else np.ones(len(cond_seq), bool)
    n, T, D = base_seq.shape
    seq = base_seq.astype(np.float64)
    shuf = shuffle_sequence(seq, "shuffled", DEFAULT_FRAME_ORDER_SEED)
    rev = seq[:, ::-1].copy()

    # ---- 0. is there anything on the temporal axis at all? ---------------------------
    unit = seq / (np.linalg.norm(seq, axis=-1, keepdims=True) + 1e-12)
    adj = np.einsum("ntd,ntd->nt", unit[:, :-1], unit[:, 1:])
    first_last = np.einsum("nd,nd->n", unit[:, 0], unit[:, -1])
    mean_raw = seq.mean(1)
    w = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    w /= np.linalg.norm(w) + 1e-12
    slope_raw = np.einsum("t,ntd->nd", w, seq)
    resid = seq - seq.mean(1, keepdims=True)
    kin = {
        "n_videos": int(n),
        "n_steps": int(T),
        "dim": int(D),
        "cos_adjacent_frames_mean": float(adj.mean()),
        "cos_adjacent_frames_min": float(adj.min()),
        "cos_first_last_frame_mean": float(first_last.mean()),
        "cos_first_last_frame_min": float(first_last.min()),
        "within_clip_frac_of_total_variance": float(
            (resid ** 2).sum() / ((seq - seq.mean((0, 1), keepdims=True)) ** 2).sum()
        ),
        "slope_over_mean_norm_ratio": float(
            (np.linalg.norm(slope_raw, axis=1) / np.linalg.norm(mean_raw, axis=1)).mean()
        ),
    }

    # ---- 1. RDM: native vs frame-shuffled, per read-out ------------------------------
    nat = temporal_readouts(seq, n_random=n_seeds, seed=seed)
    sh = temporal_readouts(shuf, n_random=n_seeds, seed=seed)
    rv = temporal_readouts(rev, n_random=n_seeds, seed=seed)
    order: Dict[str, Dict[str, Any]] = {}
    for k in nat:
        row = _order_effect(nat[k], sh[k])
        a = nat[k] / (np.linalg.norm(nat[k], axis=1, keepdims=True) + 1e-12)
        b = rv[k] / (np.linalg.norm(rv[k], axis=1, keepdims=True) + 1e-12)
        row["cos_native_vs_reversed_mean"] = float((a * b).sum(1).mean())
        row["kind"] = "closed-form"
        order[k] = row
    order = _collapse_random(order, n_seeds)

    learned_nat = learned_readouts(seq, n_seeds=n_seeds, seed=seed)
    learned_sh = learned_readouts(shuf, n_seeds=n_seeds, seed=seed)
    for pool, arms in learned_nat.items():
        rows = [_order_effect(a, b) for a, b in zip(arms, learned_sh[pool])]
        order["head:" + pool] = {
            "kind": f"random-init VideoSequenceProjector, {n_seeds} seeds",
            "cos_native_vs_shuffled_mean": float(np.mean([r["cos_native_vs_shuffled_mean"] for r in rows])),
            "cos_native_vs_shuffled_sd": float(np.std([r["cos_native_vs_shuffled_mean"] for r in rows])),
            "rdm_spearman": float(np.mean([r["rdm_spearman"] for r in rows])),
            "rdm_spearman_sd": float(np.std([r["rdm_spearman"] for r in rows])),
            "rdm_mean_abs_delta": float(np.mean([r["rdm_mean_abs_delta"] for r in rows])),
            "rdm_scale": float(np.mean([r["rdm_scale"] for r in rows])),
        }

    # ---- 2. the census LOVO metric, pooled vs temporal --------------------------------
    vtd = load_vtd(vtd_path).sort_values("video_id").reset_index(drop=True)
    X = _design_matrix(vtd)
    # every metric below is a function of inner products among the 90 videos only, so the
    # exact 90-dimensional re-coordinatisation is free of information and cheap
    red = {k: _rowspace(e) for k, e in nat.items()}
    lovo: Dict[str, Dict[str, Any]] = {}
    for k, e in red.items():
        row: Dict[str, Any] = dict(_lovo(e, X))
        # each read-out gets its *own* attribute-permutation null: the null depends on the
        # embedding's geometry, so borrowing the pooled one would be a different test
        nl = _lovo_null(e, X, n_perm=n_perm, seed=seed)
        t5 = nl.pop("_top5_null")
        nl.pop("_top1_null")
        row["top5_p_vs_attr_perm"] = float((1 + int(np.sum(t5 >= row["top5"]))) / (len(t5) + 1))
        row["top5_null_mean"] = nl["top5_null_mean"]
        row["top5_null_p95"] = nl["top5_null_p95"]
        lovo[k] = row
    for pool, arms in learned_nat.items():
        rs = [_lovo(_rowspace(a), X) for a in arms]
        lovo["head:" + pool] = {
            "top1": float(np.mean([r["top1"] for r in rs])),
            "top5": float(np.mean([r["top5"] for r in rs])),
            "top5_sd": float(np.std([r["top5"] for r in rs])),
            "mean_rank": float(np.mean([r["mean_rank"] for r in rs])),
            "top5_p_vs_attr_perm": float("nan"),
            "top5_null_mean": float("nan"),
            "top5_null_p95": float("nan"),
        }
    lovo = _collapse_random(lovo, n_seeds)
    lovo["_chance"] = {"top1": 1.0 / n, "top5": 5.0 / n}

    # ---- 3. RSA against the attribute RDMs, pooled vs temporal ------------------------
    from ...eval.census import attribute_rdms

    attr = attribute_rdms(vtd)
    rsa: Dict[str, Dict[str, float]] = {}
    for k, e in red.items():
        ev = _vec(_rdm(e))
        rsa[k] = {a: _spearman(ev, _vec(r)) for a, r in attr.items()}
    rsa = _collapse_random(rsa, n_seeds)

    # ---- 4. the directional attribute -------------------------------------------------
    direction: Dict[str, Any] = {}
    if "approaching" in vtd.columns:
        y = (vtd["approaching"].astype(str).str.lower() == "yes").to_numpy(dtype=float)
        for k, e in red.items():
            direction[k] = _directional_probe(e, y, seed=seed)
        direction = _collapse_random(direction, n_seeds)

    reliability = orientation_reliability(cond_seq, cached_flags)
    separability = _collapse_random({k: dict(gallery_separability(e)) for k, e in red.items()}, n_seeds)

    return {
        "seq_npz": str(seq_npz),
        "model_tag": meta.get("model_tag"),
        "frame_norm": meta.get("frame_norm"),
        "complete": meta.get("complete"),
        "kinematics": kin,
        "frame_order": order,
        "lovo": lovo,
        "rsa": rsa,
        "direction": direction,
        "orientation_reliability": reliability,
        "gallery_separability": separability,
        "n_perm": int(n_perm),
        "n_seeds": int(n_seeds),
        "seed": int(seed),
    }


def render_analysis(rep: Dict[str, Any]) -> str:  # noqa: D103
    k = rep["kinematics"]
    L = [
        f"# Does frame order carry anything?  {rep['model_tag']} sequence cache",
        "",
        f"90 original-orientation videos, {k['n_steps']} steps x {k['dim']} dims. "
        f"source: {rep['seq_npz']}",
        "",
        "## 0. Is there anything on the temporal axis at all?",
        "",
        f"- cos(frame_t, frame_t+1): mean {k['cos_adjacent_frames_mean']:.4f}, "
        f"min {k['cos_adjacent_frames_min']:.4f}",
        f"- cos(first frame, last frame): mean {k['cos_first_last_frame_mean']:.4f}, "
        f"min {k['cos_first_last_frame_min']:.4f}",
        f"- within-clip share of total embedding variance: "
        f"{k['within_clip_frac_of_total_variance']:.4f}",
        f"- ||temporal slope|| / ||clip mean||: {k['slope_over_mean_norm_ratio']:.4f}",
        "",
        "## 1. Native vs frame-shuffled, per read-out",
        "",
        "| read-out | kind | cos(native, shuffled) | RDM Spearman | mean abs dRDM | RDM scale |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rep["frame_order"].items():
        sd = f" +- {r['cos_native_vs_shuffled_sd']:.4f}" if "cos_native_vs_shuffled_sd" in r else ""
        rsd = f" +- {r['rdm_spearman_sd']:.4f}" if "rdm_spearman_sd" in r else ""
        L.append(
            f"| {name} | {r['kind']} | {r['cos_native_vs_shuffled_mean']:.5f}{sd} | "
            f"{r['rdm_spearman']:.5f}{rsd} | {r['rdm_mean_abs_delta']:.5f} | {r['rdm_scale']:.4f} |"
        )
    L += [
        "",
        "`mean` is the order-blind control and must read 1.00000 exactly.",
        "",
        "## 2. Leave-one-video-out attribute predictability (the census metric)",
        "",
        f"chance top-1 = {rep['lovo']['_chance']['top1']:.4f}, "
        f"top-5 = {rep['lovo']['_chance']['top5']:.4f}; "
        f"each row carries its own attribute-permutation null "
        f"({rep['n_perm']} permutations of the 90 video labels)",
        "",
        "| read-out | top-1 | top-5 | mean rank | null top-5 (95th pct) | p(top5) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rep["lovo"].items():
        if name.startswith("_"):
            continue
        sd = f" +- {r['top5_sd']:.4f}" if "top5_sd" in r else ""
        nul = (
            f"{r['top5_null_mean']:.4f} ({r['top5_null_p95']:.4f})"
            if r.get("top5_null_mean") == r.get("top5_null_mean")
            else "n/a"
        )
        L.append(
            f"| {name} | {r['top1']:.4f} | {r['top5']:.4f}{sd} | {r['mean_rank']:.2f} | {nul} | "
            f"{r['top5_p_vs_attr_perm']:.4f} |"
        )
    if rep["rsa"]:
        attrs = list(next(iter(rep["rsa"].values())).keys())
        L += ["", "## 3. RDM alignment with the behavioural attributes (Spearman)", "",
              "| read-out | " + " | ".join(attrs) + " |",
              "|---" * (len(attrs) + 1) + "|"]
        for name, row in rep["rsa"].items():
            L.append(f"| {name} | " + " | ".join(f"{row[a]:+.4f}" for a in attrs) + " |")
    if rep["direction"]:
        L += ["", "## 4. `approaching` -- the one attribute that is a statement about "
              "the direction of time", "",
              "| read-out | LOVO AUC | p (label perm) |", "|---|---|---|"]
        for name, r in rep["direction"].items():
            sd = f" +- {r['auc_sd']:.4f}" if "auc_sd" in r else ""
            L.append(f"| {name} | {r['auc']:.4f}{sd} | {r['p_label_perm']:.4f} |")
    sep = rep.get("gallery_separability") or {}
    if sep:
        L += ["", "## 5. Target geometry: how separable is the retrieval gallery?", "",
              "| read-out | mean off-diag cos | max off-diag cos | nearest-neighbour cos | "
              "effective rank |", "|---|---|---|---|---|"]
        for name, r in sep.items():
            L.append(
                f"| {name} | {r['mean_off_diagonal_cosine']:.4f} | "
                f"{r['max_off_diagonal_cosine']:.4f} | "
                f"{r['nearest_neighbour_cosine_mean']:.4f} | {r['effective_rank']:.1f} |"
            )
    rel = rep.get("orientation_reliability") or {}
    if rel.get("residual_cos_same_video_across_orientation") is not None:
        L += ["", "## 6. Is the temporal component signal or encoder noise?", "",
              f"{rel['n_videos_with_all_orientations']} videos have all four orientations "
              "cached.  A mirror does not change a clip's dynamics, so anything in the "
              "temporal axis that is a property of the event must survive one.", "",
              "| quantity | same video, across orientation | across video (baseline) |",
              "|---|---|---|",
              f"| pooled embedding cosine | "
              f"{rel['pooled_cos_same_video_across_orientation']:.4f} | "
              f"{rel['pooled_cos_across_video']:.4f} |",
              f"| order-sensitive residual cosine | "
              f"{rel['residual_cos_same_video_across_orientation']:.4f} | "
              f"{rel['residual_cos_across_video']:.4f} |",
              f"| motion profile (frame-to-frame displacement) Pearson r | "
              f"{rel['motion_profile_r_same_video_across_orientation']:.4f} | "
              f"{rel['motion_profile_r_across_video']:.4f} |"]
    return "\n".join(L) + "\n"


# ======================================================================================
# CLI
# ======================================================================================


def _cmd_build(args: argparse.Namespace) -> int:
    spec = MODEL_REGISTRY[args.model]
    build_sequence_cache(
        stim_root=args.stim_root,
        out_dir=args.out,
        spec=spec,
        device=args.device,
        n_frames=args.n_frames,
        force=args.force,
        limit=args.limit,
        synthesize_flips=args.synthesize_flips,
        fp16=args.fp16,
        tag=args.tag,
        verify_weights=not args.no_verify_weights,
        partial_ok=args.partial_ok,
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    rep = verify_against_pooled(args.seq, args.pooled)
    print(json.dumps(rep, indent=2))
    return 0 if not rep["verdict"].startswith("DISAGREES") else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    rep = run_analysis(
        seq_npz=args.seq,
        vtd_path=args.vtd,
        seed=args.seed,
        n_seeds=args.n_seeds,
        n_perm=args.n_perm,
    )
    text = render_analysis(rep)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "temporal_readout.json").write_text(json.dumps(rep, indent=2, default=str))
        (out / "temporal_readout.md").write_text(text)
        print(f"[written] {out / 'temporal_readout.md'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: D103
    ap = argparse.ArgumentParser(
        description="Frame-sequence video cache (contract C-seq).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="encode the per-frame sequence cache")
    b.add_argument("--stim-root", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("data/derived/video_emb_seq"))
    b.add_argument("--model", default="siglip2-base", choices=sorted(MODEL_REGISTRY))
    b.add_argument("--tag", default=None)
    b.add_argument("--n-frames", type=int, default=None)
    b.add_argument("--device", default="cpu")
    b.add_argument("--fp16", action="store_true")
    b.add_argument("--force", action="store_true")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--synthesize-flips", action="store_true")
    b.add_argument("--no-verify-weights", action="store_true")
    b.add_argument(
        "--partial-ok",
        action="store_true",
        help="assemble an npz from the shards present (analysis only; never train on it)",
    )
    b.set_defaults(fn=_cmd_build)

    v = sub.add_parser("verify", help="mean-pool(sequence) vs contract C")
    v.add_argument("--seq", type=Path, required=True)
    v.add_argument("--pooled", type=Path, required=True)
    v.set_defaults(fn=_cmd_verify)

    a = sub.add_parser("analyze", help="does frame order carry anything?")
    a.add_argument("--seq", type=Path, required=True)
    a.add_argument("--vtd", type=Path, required=True)
    a.add_argument("--out", type=Path, default=None)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--n-seeds", type=int, default=8)
    a.add_argument("--n-perm", type=int, default=200)
    a.set_defaults(fn=_cmd_analyze)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
