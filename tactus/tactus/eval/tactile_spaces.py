#!/usr/bin/env python
"""Build the D23 model RDMs (prereg/D23_MULTIMODAL_RSA_FROZEN.md, frozen 2026-08-29).

Five spaces at the 90-base-video grain, every definition frozen in the prereg
BEFORE any EEG correlation ran; this module only implements them:

- ``A``         SigLIP2 visual-semantic: 1 - cosine over ``base_emb (90, 768)``.
- ``B1``        tactile-adjective projection: the 30 frozen adjectives through the
                SigLIP2 *text* tower (3 templates, L2 -> mean -> L2), scored against
                ``frame_emb (360, 15, 768)``, frames then orientations averaged ->
                (90, 30) profile -> 1 - Pearson.  B1 reweights directions of the
                same tower as A -- the prereg carries this as a declared limitation.
- ``C``         affect v1: Euclidean over z-scored VTD (valence, arousal, threat, pain).
- ``material``  categorical RDM from the VTD material column (mandatory control).
- ``lowlevel``  per-video mean frame luminance, RMS contrast, mean |frame diff|
                (motion energy) from the native-orientation mp4s, z-scored, Euclidean.

The B2 (ImageBind) RDM is added by ``--imagebind-npz`` when the out-of-band
embedding exists; absent, downstream reports carry the within-tower qualifier.

QC sentinel 2 (manipulation check) is computed here and written into the
manifest: the (sharp+prickly+spiky)/3 composite must correlate positively with
VTD ``threat`` over the 90 videos under a video permutation (p < 0.05), else the
caller must STOP (the runner enforces it).

CLI::

    python -m tactus.eval.tactile_spaces --out $TACTUS_WORK/results/multimodal_rsa
    python -m tactus.eval.tactile_spaces --check-determinism
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.events import load_vtd

N_VIDEOS = 90
N_ORIENTATIONS = 4

# Frozen in the prereg -- do not edit without a dated appendum.
ADJECTIVES: List[str] = [
    "rough", "smooth", "soft", "hard", "sharp", "blunt", "sticky", "slippery",
    "wet", "dry", "warm", "cold", "fuzzy", "furry", "silky", "prickly",
    "spiky", "bumpy", "grainy", "slimy", "greasy", "rubbery", "firm",
    "squishy", "spongy", "coarse", "springy", "heavy", "light", "textured",
]
TEMPLATES = ("{adj}", "a {adj} surface", "something that feels {adj} to touch")
AFFECT_COLS = ("valence", "arousal", "threat", "pain")
MANIPULATION_COMPOSITE = ("sharp", "prickly", "spiky")
SIGLIP_ID = "google/siglip2-base-patch16-224"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rdm_cosine(x: np.ndarray) -> np.ndarray:
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    return (1.0 - x @ x.T).astype(np.float64)


def _rdm_pearson(x: np.ndarray) -> np.ndarray:
    return (1.0 - np.corrcoef(x)).astype(np.float64)


def _rdm_euclid_z(x: np.ndarray) -> np.ndarray:
    z = (x - x.mean(axis=0)) / x.std(axis=0).clip(1e-12)
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2)


def _rdm_categorical(labels: pd.Series) -> np.ndarray:
    a = labels.to_numpy()
    return (a[:, None] != a[None, :]).astype(np.float64)


# --------------------------------------------------------------------------- #
# spaces
# --------------------------------------------------------------------------- #
def space_A(emb_npz: Path) -> np.ndarray:
    base = np.load(emb_npz)["base_emb"]
    assert base.shape[0] == N_VIDEOS, base.shape
    return _rdm_cosine(base.astype(np.float64))


def adjective_embeddings() -> np.ndarray:
    """(30, 768) SigLIP2 text-tower embeddings, template-ensembled, L2 rows."""
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.manual_seed(0)
    model = AutoModel.from_pretrained(SIGLIP_ID)
    model.eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    out = np.zeros((len(ADJECTIVES), model.config.text_config.hidden_size))
    with torch.no_grad():
        for i, adj in enumerate(ADJECTIVES):
            prompts = [t.format(adj=adj) for t in TEMPLATES]
            tok = proc(text=prompts, padding="max_length", return_tensors="pt")
            feats = model.get_text_features(**tok).double().numpy()
            feats /= np.linalg.norm(feats, axis=1, keepdims=True)
            v = feats.mean(axis=0)
            out[i] = v / np.linalg.norm(v)
    return out


def space_B1_profile(frames_npz: Path, adj_emb: np.ndarray) -> np.ndarray:
    """(90, 30) adjective profile: cos over frames, mean frames, mean orientations."""
    fr = np.load(frames_npz)["frame_emb"].astype(np.float64)  # (360, 15, 768)
    assert fr.shape[0] == N_VIDEOS * N_ORIENTATIONS, fr.shape
    fr /= np.linalg.norm(fr, axis=2, keepdims=True).clip(1e-12)
    scores = np.einsum("cfd,ad->cfa", fr, adj_emb).mean(axis=1)  # (360, 30)
    return scores.reshape(N_VIDEOS, N_ORIENTATIONS, -1).mean(axis=1)


def space_C(vtd: pd.DataFrame) -> np.ndarray:
    return _rdm_euclid_z(vtd[list(AFFECT_COLS)].to_numpy(dtype=np.float64))


def lowlevel_features(stim_root: Path) -> np.ndarray:
    """(90, 3): mean luminance, RMS contrast, motion energy from native mp4s."""
    import cv2

    from ..models.video.encode import discover_orientation_dirs

    native = discover_orientation_dirs(stim_root)[0]
    files = sorted(native.glob("*.mp4"))
    if len(files) != N_VIDEOS:
        raise RuntimeError(f"expected {N_VIDEOS} native mp4s, found {len(files)} in {native}")

    from ..data.events import parse_stim

    by_vid: Dict[int, Path] = {}
    for f in files:
        vid = parse_stim(str(f)).video_id
        if vid in by_vid:
            raise RuntimeError(f"duplicate video_id {vid}: {f} and {by_vid[vid]}")
        by_vid[vid] = f
    if sorted(by_vid) != list(range(1, N_VIDEOS + 1)):
        raise RuntimeError("native orientation dir does not cover video_id 1..90 exactly")

    out = np.zeros((N_VIDEOS, 3))
    for vid in range(1, N_VIDEOS + 1):
        cap = cv2.VideoCapture(str(by_vid[vid]))
        grays = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0)
        cap.release()
        if len(grays) < 2:
            raise RuntimeError(f"decoded {len(grays)} frames from {by_vid[vid]}")
        g = np.stack(grays)
        out[vid - 1, 0] = g.mean()
        out[vid - 1, 1] = g.std(axis=(1, 2)).mean()
        out[vid - 1, 2] = np.abs(np.diff(g, axis=0)).mean()
    return out


# --------------------------------------------------------------------------- #
# QC sentinel 2: manipulation check
# --------------------------------------------------------------------------- #
def manipulation_check(profile: np.ndarray, vtd: pd.DataFrame,
                       n_perm: int = 5000, seed: int = 0) -> Dict[str, float]:
    from scipy import stats

    idx = [ADJECTIVES.index(a) for a in MANIPULATION_COMPOSITE]
    comp = profile[:, idx].mean(axis=1)
    threat = vtd["threat"].to_numpy(dtype=np.float64)
    r_obs = float(stats.spearmanr(comp, threat).statistic)
    rng = np.random.default_rng(seed)
    null = np.array([
        stats.spearmanr(comp, rng.permutation(threat)).statistic for _ in range(n_perm)
    ])
    p = float((1 + np.sum(null >= r_obs)) / (1 + n_perm))  # one-sided: positive
    return {"r": r_obs, "p": p, "n_perm": n_perm, "passed": bool(r_obs > 0 and p < 0.05)}


# --------------------------------------------------------------------------- #
# build + manifest
# --------------------------------------------------------------------------- #
def build(out_dir: Path, *, emb_dir: Path, vtd_csv: Path, stim_root: Path,
          imagebind_npz: Optional[Path] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    vtd = load_vtd(vtd_csv)
    assert len(vtd) == N_VIDEOS, f"VTD rows: {len(vtd)}"
    vtd = vtd.sort_values("video_id").reset_index(drop=True) if "video_id" in vtd else vtd

    adj = adjective_embeddings()
    profile = space_B1_profile(emb_dir / "siglip2-base-frames.npz", adj)
    low = lowlevel_features(stim_root)

    rdms: Dict[str, np.ndarray] = {
        "A": space_A(emb_dir / "siglip2-base.npz"),
        "B1": _rdm_pearson(profile),
        "C": space_C(vtd),
        "material": _rdm_categorical(vtd["material"]),
        "lowlevel": _rdm_euclid_z(low),
    }
    if imagebind_npz is not None:
        ib = np.load(imagebind_npz)["frame_emb"].astype(np.float64)
        ib /= np.linalg.norm(ib, axis=-1, keepdims=True).clip(1e-12)
        pooled = ib.mean(axis=1)
        pooled = pooled.reshape(N_VIDEOS, N_ORIENTATIONS, -1).mean(axis=1) \
            if pooled.shape[0] == N_VIDEOS * N_ORIENTATIONS else pooled
        rdms["B2"] = _rdm_cosine(pooled)

    qc = manipulation_check(profile, vtd)

    npz_path = out_dir / "model_rdms.npz"
    np.savez(npz_path, adjective_profile=profile, lowlevel_features=low,
             adjective_names=np.array(ADJECTIVES),
             **{f"rdm_{k}": v for k, v in rdms.items()})
    manifest = {
        "frozen_prereg": "prereg/D23_MULTIMODAL_RSA_FROZEN.md",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "siglip2-base.npz": _sha256(emb_dir / "siglip2-base.npz"),
            "siglip2-base-frames.npz": _sha256(emb_dir / "siglip2-base-frames.npz"),
            "VTD.csv": _sha256(vtd_csv),
        },
        "spaces": sorted(rdms),
        "b2_present": "B2" in rdms,
        "adjectives": ADJECTIVES,
        "templates": list(TEMPLATES),
        "qc_manipulation_check": qc,
        "output_sha256": _sha256(npz_path),
    }
    (out_dir / "spaces_manifest.json").write_text(json.dumps(manifest, indent=2))
    return npz_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    work = Path(os.environ.get("TACTUS_WORK", "/projects/EEG-foundation-model/tactus_work"))
    bids = Path(os.environ.get("TACTUS_BIDS_ROOT", "/projects/EEG-foundation-model/ds005662"))
    ap.add_argument("--out", type=Path, default=work / "results" / "multimodal_rsa")
    ap.add_argument("--emb-dir", type=Path, default=work / "derived" / "video_emb")
    ap.add_argument("--vtd", type=Path, default=bids / "code" / "analysis" / "VTD.csv")
    ap.add_argument("--stim-root", type=Path,
                    default=bids / "code" / "experiment_files" / "stimuli")
    ap.add_argument("--imagebind-npz", type=Path, default=None)
    ap.add_argument("--check-determinism", action="store_true",
                    help="build twice into scratch subdirs and require byte-identical npz")
    args = ap.parse_args(argv)

    if args.check_determinism:
        kw = dict(emb_dir=args.emb_dir, vtd_csv=args.vtd, stim_root=args.stim_root,
                  imagebind_npz=args.imagebind_npz)
        p1 = build(args.out / "_det_a", **kw)
        p2 = build(args.out / "_det_b", **kw)
        z1, z2 = np.load(p1), np.load(p2)
        same = set(z1.files) == set(z2.files) and all(
            np.array_equal(z1[k], z2[k]) for k in z1.files)
        print(f"determinism: {'PASS (bit-identical arrays)' if same else 'FAIL'}")
        return 0 if same else 1

    npz = build(args.out, emb_dir=args.emb_dir, vtd_csv=args.vtd,
                stim_root=args.stim_root, imagebind_npz=args.imagebind_npz)
    man = json.loads((args.out / "spaces_manifest.json").read_text())
    qc = man["qc_manipulation_check"]
    print(f"wrote {npz}")
    print(f"QC sentinel 2 (sharp/prickly/spiky vs threat): r={qc['r']:.3f} "
          f"p={qc['p']:.4f} -> {'PASS' if qc['passed'] else 'FAIL -- STOP'}")
    return 0 if qc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
