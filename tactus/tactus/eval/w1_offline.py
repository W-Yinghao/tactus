#!/usr/bin/env python
"""W1 offline calibration (EXPLORATION_PROGRAM_v2 §2, §3 E1a-i, §3 补两项).

Three computations on EXISTING fold outputs, no training:

1. **E1a-i** -- InfoNCE (projector trained) vs ProtoNCE (projector frozen at
   init, D24 finding) on within_subject folds vf00-vf03, as the program's
   statistics constitution demands: **per-video k=1 paired differences** over
   the 18 held-out videos per fold (72 exchangeable units), with the per-video
   SD and the implied MDD.  Also the k=4 pseudo-trial version for reference.
   The single-trial 18-way top-1 recomputed here is verified against each
   fold's booked ``test/video/g18/top1`` before anything else is reported.

2. **k calibration curve** -- ProtoNCE vf00-vf03, pseudo-trial k in
   {1, 2, 4, 8}, 5 resamples: the endpoint's dependence on averaging.

3. **k=1 noise ceiling** -- ``retrieval_noise_ceiling(k=1, k_gallery=7)``, the
   documented closest approximation to a single-trial ceiling, beside the
   k=4/k=4 reference, so the k=1 headroom is known for the first time.

Writes ``$TACTUS_WORK/results/w1_offline/{e1a_pervideo.csv, e1a_summary.json,
k_calibration.csv, k1_ceiling.csv}``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .noise_ceiling import retrieval_noise_ceiling
from .retrieval import build_pseudo_trials

FOLDS = ("vf00", "vf01", "vf02", "vf03")
ARMS = {"infonce": "nice_infonce", "protonce": "nice_protonce__subj80"}
KS = (1, 2, 4, 8)


def _work() -> Path:
    return Path(os.environ.get("TACTUS_WORK", "/projects/EEG-foundation-model/tactus_work"))


def _fold(run: str, fold: str) -> Path:
    return _work() / "results" / "runs" / run / "folds" / fold


def _load(run: str, fold: str) -> Dict[str, np.ndarray]:
    z = np.load(_fold(run, fold) / "test_embeddings.npz")
    return {k: z[k] for k in z.files}


def _booked_top1(run: str, fold: str) -> float:
    df = pd.read_csv(_work() / "results" / "runs" / run / "summary_folds.csv")
    row = df[df.fold_key == fold]
    return float(row["test/video/g18/top1"].iloc[0])


def _top1_per_trial(z: np.ndarray, gal: np.ndarray, vids: np.ndarray, gal_vids: np.ndarray) -> np.ndarray:
    sim = z @ gal.T
    pred = gal_vids[sim.argmax(axis=1)]
    return (pred == vids).astype(float)


def e1a_pervideo(out_dir: Path) -> None:
    rows: List[dict] = []
    for fold in FOLDS:
        per_arm: Dict[str, Dict[int, float]] = {}
        per_arm_k4: Dict[str, Dict[int, float]] = {}
        for arm, run in ARMS.items():
            d = _load(run, fold)
            z, gal = d["z_eeg"].astype(np.float64), d["z_vid_video"].astype(np.float64)
            vids, gal_vids = d["video_id"].astype(int), d["gallery_video_ids"].astype(int)
            hit = _top1_per_trial(z, gal, vids, gal_vids)
            booked = _booked_top1(run, fold)
            if abs(hit.mean() - booked) > 2e-3:
                raise RuntimeError(f"{run}/{fold}: recomputed k=1 top1 {hit.mean():.4f} != booked "
                                   f"{booked:.4f}; the eval convention was not reproduced -- STOP")
            per_arm[arm] = {int(v): float(hit[vids == v].mean()) for v in gal_vids}
            # k=4 pseudo trials per subject, seed 0
            za, items, subs = build_pseudo_trials(z, vids, k=4, subject_ids=d["subject_id"],
                                                  rng=np.random.default_rng(0))[:3]
            hit4 = _top1_per_trial(np.asarray(za), gal, np.asarray(items).astype(int), gal_vids)
            per_arm_k4[arm] = {int(v): float(hit4[np.asarray(items).astype(int) == v].mean())
                               for v in gal_vids}
        for v in per_arm["infonce"]:
            rows.append({"fold": fold, "video_id": v,
                         "k1_infonce": per_arm["infonce"][v], "k1_protonce": per_arm["protonce"][v],
                         "k4_infonce": per_arm_k4["infonce"][v], "k4_protonce": per_arm_k4["protonce"][v]})
    df = pd.DataFrame(rows)
    df["d_k1"] = df.k1_infonce - df.k1_protonce
    df["d_k4"] = df.k4_infonce - df.k4_protonce
    df.to_csv(out_dir / "e1a_pervideo.csv", index=False)

    from scipy import stats
    summ = {}
    for col in ("d_k1", "d_k4"):
        d = df[col].to_numpy()
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        summ[col] = {
            "n_videos": int(n), "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)),
            "se": float(se), "mdd_paired_80pct": float((1.96 + 0.84) * se),
            "t": float(d.mean() / se), "p_t": float(2 * stats.t.sf(abs(d.mean() / se), n - 1)),
            "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        }
    summ["arm_means"] = {c: float(df[c].mean()) for c in ("k1_infonce", "k1_protonce", "k4_infonce", "k4_protonce")}
    (out_dir / "e1a_summary.json").write_text(json.dumps(summ, indent=2))
    print("E1a-i:", json.dumps(summ, indent=1))


def k_calibration(out_dir: Path) -> None:
    rows = []
    for fold in FOLDS:
        d = _load(ARMS["protonce"], fold)
        z, gal = d["z_eeg"].astype(np.float64), d["z_vid_video"].astype(np.float64)
        vids, gal_vids = d["video_id"].astype(int), d["gallery_video_ids"].astype(int)
        for k in KS:
            accs = []
            for r in range(5):
                if k == 1:
                    accs.append(_top1_per_trial(z, gal, vids, gal_vids).mean())
                    break
                za, items, _ = build_pseudo_trials(z, vids, k=k, subject_ids=d["subject_id"],
                                                   rng=np.random.default_rng(r))[:3]
                accs.append(_top1_per_trial(np.asarray(za), gal, np.asarray(items).astype(int), gal_vids).mean())
            rows.append({"fold": fold, "k": k, "top1": float(np.mean(accs)), "n_resamples": len(accs)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "k_calibration.csv", index=False)
    print("k calibration (mean over folds):")
    print(df.groupby("k").top1.agg(["mean", "std"]).to_string())


def k1_ceiling(out_dir: Path) -> None:
    rows = []
    for fold in FOLDS:
        d = _load(ARMS["protonce"], fold)
        for k, kg in ((1, 7), (1, 1), (4, 4)):
            df = retrieval_noise_ceiling(
                d["z_eeg"], d["video_id"], d["subject_id"], k=k, k_gallery=kg,
                gallery_sizes=[18], n_gallery_subjects=10, n_gallery_draws=20, seed=0,
            )
            df["fold"], df["k"], df["k_gallery"] = fold, k, kg
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_dir / "k1_ceiling.csv", index=False)
    cols = [c for c in out.columns if c in ("fold", "k", "k_gallery", "scope", "gallery", "top1", "ceiling", "ceiling_sd")]
    print("ceilings:")
    print(out[cols].to_string() if cols else out.head(12).to_string())


def main() -> int:
    out_dir = _work() / "results" / "w1_offline"
    out_dir.mkdir(parents=True, exist_ok=True)
    e1a_pervideo(out_dir)
    k_calibration(out_dir)
    k1_ceiling(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
