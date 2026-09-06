#!/usr/bin/env python
"""W1 endpoints (prereg/W1_EXPLORATION_FROZEN.md) over the completed fleet.

Per arm x seed x fold: per-video k=1 18-way top-1 (recomputed from
``test_embeddings.npz`` with the trainer's own uncentred cosine, verified against
the booked fold number), the material/low-level-partialled affect probe and the
visual probe on per-video EEG prototypes.  Then the two confirmatory contrasts
(P1, P2a, P2b) with their paired-video statistics, and the exploratory table.
Coverage is enforced: an arm with a missing (fold, seed) is reported as
incomplete and excluded from confirmatory rows -- never silently averaged.

    python -m tactus.eval.w1_analysis            # writes results/w1/{*.csv, W1_RESULTS.json}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..data.events import load_vtd

FOLDS = ("vf00", "vf01", "vf02", "vf03")
SEEDS = (0, 1, 2)
N_PERM = 5000
MDD_K1 = 0.015

ARMS: Dict[str, Dict[int, str]] = {
    # arm -> seed -> run directory
    "vid_protonce": {0: "nice_protonce__subj80", 1: "nice_protonce_w1_vid_s1", 2: "nice_protonce_w1_vid_s2"},
    "vid_codebook": {s: f"w1_codebook_ce_s{s}" for s in SEEDS},
    "capfull": {s: f"nice_protonce_w1_capfull_s{s}" for s in SEEDS},
    "capvis": {s: f"nice_protonce_w1_capvis_s{s}" for s in SEEDS},
    "capblind": {s: f"nice_protonce_w1_capblind_s{s}" for s in SEEDS},
    "mat_supcon": {s: f"w1_supcon_material_s{s}" for s in SEEDS},
    "aff_rnc": {s: f"w1_rnc_affect_s{s}" for s in SEEDS},
    **{f"capvis_w{w}": {s: f"nice_protonce_w1_capvis_w{w}_s{s}" for s in SEEDS} for w in (0, 1, 2)},
    **{f"aff_rnc_w{w}": {s: f"w1_rnc_affect_w{w}_s{s}" for s in SEEDS} for w in (0, 1, 2)},
}


def _work() -> Path:
    return Path(os.environ.get("TACTUS_WORK", "/projects/EEG-foundation-model/tactus_work"))


def _runs() -> Path:
    return _work() / "results" / "runs"


# --------------------------------------------------------------------------- #
# stimulus-side covariates (material dummies + low-level features) and targets
# --------------------------------------------------------------------------- #
def _stimulus_tables() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    vtd = load_vtd(Path(os.environ.get("TACTUS_BIDS_ROOT", "/projects/EEG-foundation-model/ds005662"))
                   / "code" / "analysis" / "VTD.csv").sort_values("video_id").reset_index(drop=True)
    spaces = np.load(_work() / "results" / "multimodal_rsa" / "model_rdms.npz")
    low = spaces["lowlevel_features"]                                   # (90, 3)
    mat = pd.get_dummies(vtd["material"].astype(str)).to_numpy(float)   # (90, 8)
    controls = np.column_stack([mat[:, :-1], (low - low.mean(0)) / low.std(0)])
    aff = vtd[["valence", "arousal", "threat"]].to_numpy(float)
    aff = (aff - aff.mean(0)) / aff.std(0)
    base = np.load(_work() / "derived" / "video_emb" / "siglip2-base.npz")["base_emb"].astype(float)
    base = base - base.mean(0)
    u, s_, vt = np.linalg.svd(base, full_matrices=False)
    vis_pcs = u[:, :5] * s_[:5]                                          # (90, 5)
    return vtd, controls, np.column_stack([aff, vis_pcs])


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


# --------------------------------------------------------------------------- #
# per-fold quantities
# --------------------------------------------------------------------------- #
def _fold_dir(run: str, fold: str) -> Path:
    return _runs() / run / "folds" / fold


def _booked(run: str, fold: str, key: str) -> Optional[float]:
    p = _runs() / run / "summary_folds.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    r = df[df.fold_key == fold]
    return float(r[key].iloc[0]) if len(r) else None


def fold_quantities(run: str, fold: str, targets90: np.ndarray, controls90: np.ndarray) -> Optional[dict]:
    p = _fold_dir(run, fold) / "test_embeddings.npz"
    if not p.exists():
        return None
    z = np.load(p)
    zeeg, gal = z["z_eeg"].astype(float), z["z_vid_video"].astype(float)
    vids, gal_vids = z["video_id"].astype(int), z["gallery_video_ids"].astype(int)
    subs = z["subject_id"].astype(int)
    hit = (gal_vids[(zeeg @ gal.T).argmax(1)] == vids).astype(float)
    booked = _booked(run, fold, "test/video/g18/top1")
    if booked is not None and abs(hit.mean() - booked) > 2e-3:
        raise RuntimeError(f"{run}/{fold}: recomputed k=1 {hit.mean():.4f} != booked {booked:.4f}")
    per_video_k1 = {int(v): float(hit[vids == v].mean()) for v in gal_vids}
    # per-video EEG prototypes, averaged over subjects (each subject's prototype L2-normalised)
    protos = []
    for v in gal_vids:
        ps = []
        for s in np.unique(subs):
            m = (vids == v) & (subs == s)
            if m.any():
                q = zeeg[m].mean(0)
                ps.append(q / max(np.linalg.norm(q), 1e-12))
        protos.append(np.mean(ps, axis=0))
    return {"gal_vids": gal_vids, "k1": per_video_k1, "protos": np.stack(protos),
            "code_commit": _commit(run, fold)}


def _commit(run: str, fold: str) -> str:
    p = _runs() / run / "summary_folds.csv"
    if not p.exists():
        return ""
    df = pd.read_csv(p)
    r = df[df.fold_key == fold]
    return str(r["code_commit"].iloc[0]) if len(r) and "code_commit" in r else ""


def lovo_probe(protos: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Leave-one-video-out ridge from prototypes to targets; returns predictions."""
    n = protos.shape[0]
    pred = np.zeros_like(y, dtype=float)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = protos[tr].mean(0), protos[tr].std(0).clip(1e-12)
        Xtr, Xte = (protos[tr] - mu) / sd, (protos[i] - mu) / sd
        ym = y[tr].mean(0)
        w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - ym))
        pred[i] = Xte @ w + ym
    return pred


# --------------------------------------------------------------------------- #
# arm-level assembly
# --------------------------------------------------------------------------- #
def collect(targets90: np.ndarray, controls90: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    k1_rows, probe_rows, coverage = [], [], {}
    for arm, seeds in ARMS.items():
        n_ok = 0
        commits = set()
        for seed, run in seeds.items():
            for fold in FOLDS:
                q = fold_quantities(run, fold, targets90, controls90)
                if q is None:
                    continue
                n_ok += 1
                commits.add(q["code_commit"])
                for v, acc in q["k1"].items():
                    k1_rows.append({"arm": arm, "seed": seed, "fold": fold, "video_id": v, "k1": acc})
                gv = q["gal_vids"]
                y = targets90[gv - 1]                       # (18, 8): 3 affect + 5 visual PCs
                c = controls90[gv - 1]
                pred = lovo_probe(q["protos"], y)
                for j, v in enumerate(gv):
                    probe_rows.append({"arm": arm, "seed": seed, "fold": fold, "video_id": int(v),
                                       **{f"pred_{k}": pred[j, k] for k in range(8)},
                                       **{f"true_{k}": y[j, k] for k in range(8)},
                                       **{f"ctrl_{k}": c[j, k] for k in range(c.shape[1])}})
        coverage[arm] = {"units": n_ok, "expected": 4 * len(seeds), "complete": n_ok == 4 * len(seeds),
                         "commits": sorted(commits)}
    return pd.DataFrame(k1_rows), pd.DataFrame(probe_rows), coverage


def per_video_k1(k1: pd.DataFrame, arm: str) -> pd.Series:
    """Seed-averaged per-video k=1 (index video_id, 72 values)."""
    return k1[k1.arm == arm].groupby("video_id").k1.mean()


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    boots = [np.random.default_rng(i).choice(d, n).mean() for i in range(5000)]
    return {"n": int(n), "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_t": float(2 * stats.t.sf(abs(d.mean() / se), n - 1)) if se > 0 else 1.0,
            "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0}


def probe_rho(pr: pd.DataFrame, arm: str, which: str, partial: bool = True) -> Tuple[float, np.ndarray]:
    """Pooled (72-video) Spearman between LOVO prediction and truth for the affect
    block (mean over 3 targets) or the visual block (RDM Spearman over 5 PCs)."""
    g = pr[pr.arm == arm].groupby("video_id").mean(numeric_only=True)   # seed/fold averaged per video
    ctrl = g[[c for c in g.columns if c.startswith("ctrl_")]].to_numpy()
    if which == "affect":
        rhos = []
        for k in range(3):
            p, t = g[f"pred_{k}"].to_numpy(), g[f"true_{k}"].to_numpy()
            if partial:
                p, t = _residualize(p, ctrl), _residualize(t, ctrl)
            rhos.append(stats.spearmanr(p, t).statistic)
        return float(np.mean(rhos)), g.index.to_numpy()
    P = g[[f"pred_{k}" for k in range(3, 8)]].to_numpy()
    T = g[[f"true_{k}" for k in range(3, 8)]].to_numpy()
    if partial:
        P = np.column_stack([_residualize(P[:, k], ctrl) for k in range(P.shape[1])])
        T = np.column_stack([_residualize(T[:, k], ctrl) for k in range(T.shape[1])])
    iu = np.triu_indices(len(g), 1)
    rp = 1 - np.corrcoef(P)[iu]
    rt = 1 - np.corrcoef(T)[iu]
    return float(stats.spearmanr(rp, rt).statistic), g.index.to_numpy()


def perm_p_rho_diff(pr: pd.DataFrame, arm_a: str, arm_b: str, which: str, n_perm: int = N_PERM) -> dict:
    """Video-level permutation: shuffle target rows (same permutation for both arms)."""
    rho_a, vids = probe_rho(pr, arm_a, which)
    rho_b, _ = probe_rho(pr, arm_b, which)
    obs = rho_a - rho_b
    rng = np.random.default_rng(0)
    ga = pr[pr.arm == arm_a].groupby("video_id").mean(numeric_only=True)
    gb = pr[pr.arm == arm_b].groupby("video_id").mean(numeric_only=True)
    tcols = [f"true_{k}" for k in range(8)]
    null = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(ga))
        pa, pb = ga.copy(), gb.copy()
        pa[tcols] = ga[tcols].to_numpy()[perm]
        pb[tcols] = gb[tcols].to_numpy()[perm]
        pa["arm"], pb["arm"] = "A", "B"
        both = pd.concat([pa.reset_index(), pb.reset_index()])
        null[i] = probe_rho(both, "A", which)[0] - probe_rho(both, "B", which)[0]
    return {"rho_a": rho_a, "rho_b": rho_b, "diff": obs,
            "p_perm": float((1 + np.sum(null >= obs)) / (1 + n_perm))}


def video_bootstrap_rho(pr: pd.DataFrame, arm: str, which: str, n_boot: int = 2000) -> Tuple[float, float]:
    g = pr[pr.arm == arm].groupby("video_id").mean(numeric_only=True).reset_index()
    vals = []
    for i in range(n_boot):
        take = np.random.default_rng(i).integers(0, len(g), len(g))
        sub = g.iloc[take].copy()
        sub["video_id"] = np.arange(len(sub))
        sub["arm"] = arm
        vals.append(probe_rho(sub, arm, which)[0])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main() -> int:
    out = _work() / "results" / "w1"
    out.mkdir(parents=True, exist_ok=True)
    vtd, controls90, targets90 = _stimulus_tables()
    k1, pr, coverage = collect(targets90, controls90)
    k1.to_csv(out / "per_video_k1.csv", index=False)
    pr.to_csv(out / "probe_rows.csv", index=False)

    res: dict = {"coverage": coverage, "mdd_k1": MDD_K1}
    complete = {a for a, c in coverage.items() if c["complete"]}

    # arm table (exploratory): seed-averaged mean k=1, booked k=4 not recomputed here
    res["arm_k1"] = {a: float(per_video_k1(k1, a).mean()) for a in ARMS if a in set(k1.arm)}

    # P1
    if {"vid_codebook", "vid_protonce"} <= complete:
        p1 = paired(per_video_k1(k1, "vid_codebook"), per_video_k1(k1, "vid_protonce"))
        d = p1["mean_diff"]
        p1["grid"] = ("frozen projector costs retrieval" if d > MDD_K1 and p1["ci95"][0] > 0 else
                      "live codebook hurts (frozen codebook regularizes)" if d < -MDD_K1 and p1["ci95"][1] < 0 else
                      f"no detectable cost at MDD {MDD_K1:.3f}")
        res["P1"] = p1
    # P2a
    if {"aff_rnc", "capvis"} <= complete:
        res["P2a"] = perm_p_rho_diff(pr, "aff_rnc", "capvis", "affect")
        res["P2a"]["ci_aff_rnc"] = video_bootstrap_rho(pr, "aff_rnc", "affect")
        res["P2a"]["ci_capvis"] = video_bootstrap_rho(pr, "capvis", "affect")
    # P2b
    if {"capvis_w0", "capvis_w2"} <= complete:
        res["P2b_visual_early_minus_late"] = paired(per_video_k1(k1, "capvis_w0"), per_video_k1(k1, "capvis_w2"))
    if {"aff_rnc_w0", "aff_rnc_w2"} <= complete:
        r_late, _ = probe_rho(pr, "aff_rnc_w2", "affect")
        r_early, _ = probe_rho(pr, "aff_rnc_w0", "affect")
        res["P2b_affect_late_minus_early"] = {
            "rho_late": r_late, "rho_early": r_early, "diff": r_late - r_early,
            "ci_late": video_bootstrap_rho(pr, "aff_rnc_w2", "affect"),
            "ci_early": video_bootstrap_rho(pr, "aff_rnc_w0", "affect")}
    # exploratory probes per arm
    res["probes"] = {a: {"affect_rho_partial": probe_rho(pr, a, "affect")[0],
                         "visual_rho_partial": probe_rho(pr, a, "visual")[0]}
                     for a in ARMS if a in set(pr.arm)}
    # exploratory pairwise k=1 vs T-vid ProtoNCE
    if "vid_protonce" in complete:
        base = per_video_k1(k1, "vid_protonce")
        res["k1_vs_vid_protonce"] = {a: paired(per_video_k1(k1, a), base)
                                     for a in complete if a != "vid_protonce"}
    (out / "W1_RESULTS.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in res if k in ("coverage", "P1", "P2a", "P2b_visual_early_minus_late",
                                                       "P2b_affect_late_minus_early", "arm_k1")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
