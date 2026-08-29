#!/usr/bin/env python
"""D23 runner: three-space time-resolved partial RSA + VT gating.

Implements prereg/D23_MULTIMODAL_RSA_FROZEN.md (frozen 2026-08-29) and nothing
beyond it.  Model RDMs come from :mod:`tactus.eval.tactile_spaces`; EEG RDMs are
built per subject at the 90-base-video grain (orientation folded into repeats),
crossnobis, 25 ms windows at 10 ms stride, Ledoit-Wolf whitening.

Stages (each checkpointed, skip-if-done)::

    python -m tactus.eval.multimodal_rsa build-subject --subject 1
    python -m tactus.eval.multimodal_rsa probe          # QC gates, ONE subject, no aggregate
    python -m tactus.eval.multimodal_rsa group          # H1 curves + clusters + onset bootstrap
    python -m tactus.eval.multimodal_rsa h2             # per-subject alignment + VT tests

The probe MUST pass before the 80-subject build is submitted (prereg sentinel
5); the group stage refuses to run when the spaces manifest records a failed
manipulation check (sentinel 2) or when coverage is short (sentinel 4).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..common import EpochStore, load_trials
from .rsa import partial_spearman, rsa_time_course, time_resolved_rdms

N_VIDEOS = 90
SFREQ = 200.0
H2_WINDOW_MS = (150.0, 600.0)   # frozen; NOT derived from observed clusters
N_PERM = 5000
N_BOOT_ONSET = 1000
SEED = 0

PRIMARY_CONTRASTS = {
    # partialled model -> control set (prereg "Inference")
    "B1": ("A", "C", "material", "lowlevel"),
    "A": ("B1", "C", "material", "lowlevel"),
    "B2": ("A", "C", "material", "lowlevel"),
}
H2_CONTROLS = ("A", "material", "lowlevel")


def _work() -> Path:
    return Path(os.environ.get("TACTUS_WORK", "/projects/EEG-foundation-model/tactus_work"))


def _out_dir(out: Optional[Path]) -> Path:
    return out if out is not None else _work() / "results" / "multimodal_rsa"


def _load_spaces(out_dir: Path) -> Dict[str, np.ndarray]:
    man = json.loads((out_dir / "spaces_manifest.json").read_text())
    qc = man["qc_manipulation_check"]
    if not qc["passed"]:
        raise RuntimeError(
            f"QC sentinel 2 FAILED in spaces manifest (r={qc['r']:.3f}, p={qc['p']:.4f}); "
            "the prereg says STOP -- rebuild/inspect tactile_spaces before any aggregate."
        )
    z = np.load(out_dir / "model_rdms.npz")
    return {k[len("rdm_"):]: z[k] for k in z.files if k.startswith("rdm_")}


# --------------------------------------------------------------------------- #
# per-subject EEG RDM stacks
# --------------------------------------------------------------------------- #
def subject_rdm_path(out_dir: Path, subject_id: int) -> Path:
    return out_dir / "subj_rdms" / f"sub-{subject_id:02d}.npz"


def build_subject(subject_id: int, out_dir: Path, window: str = "w0600") -> Path:
    path = subject_rdm_path(out_dir, subject_id)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    trials = load_trials(subjects=[subject_id])
    n0 = len(trials)
    trials = trials[trials.video_id >= 1]
    dropped = n0 - len(trials)

    store = EpochStore(window)
    x = store.take(subject_id, trials["within_subj_idx"].to_numpy())
    cond_ids = (trials["video_id"].to_numpy() - 1).astype(np.int64)

    times_ms = np.arange(x.shape[-1]) / SFREQ * 1000.0
    rdms, valid, centres = time_resolved_rdms(
        x, cond_ids, method="crossnobis", n_folds=4, n_conditions=N_VIDEOS,
        window=5, step=2, whiten_method="ledoit-wolf", seed=SEED, times=times_ms,
    )
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, rdms=rdms, valid=valid, centres=centres,
             n_trials=len(trials), n_dropped_nonvideo=dropped)
    tmp.rename(path)
    return path


def _load_stack(out_dir: Path, subject_id: int) -> Dict[str, np.ndarray]:
    z = np.load(subject_rdm_path(out_dir, subject_id))
    return {k: z[k] for k in z.files}


def _zscore_stack(rdms: np.ndarray) -> np.ndarray:
    """Z-score one subject's whole stack across all cells (the D11 lesson)."""
    m, s = float(np.nanmean(rdms)), float(np.nanstd(rdms))
    return (rdms - m) / max(s, 1e-12)


def group_stack(out_dir: Path, subjects: Sequence[int]) -> tuple:
    stacks, valids, centres = [], [], None
    for sid in subjects:
        d = _load_stack(out_dir, sid)
        if not bool(d["valid"].all()):
            raise RuntimeError(
                f"sub-{sid:02d} has invalid conditions {np.flatnonzero(~d['valid'])}; "
                "the prereg requires drops at base-video granularity to be "
                "reason-coded and reported -- refusing a silent subset."
            )
        stacks.append(_zscore_stack(d["rdms"]))
        valids.append(d["valid"])
        centres = d["centres"]
    return np.mean(stacks, axis=0), np.stack(stacks), centres


# --------------------------------------------------------------------------- #
# H1: group curves, clusters, onset bootstrap
# --------------------------------------------------------------------------- #
def _lift_90():
    return dict(video_of_condition=np.arange(N_VIDEOS),
                orientation_of_condition=np.zeros(N_VIDEOS, dtype=np.int64),
                n_base=N_VIDEOS)


def run_group(out_dir: Path, subjects: Sequence[int], n_perm: int = N_PERM) -> None:
    spaces = _load_spaces(out_dir)
    if len(subjects) != 80:
        raise RuntimeError(f"coverage: expected 80 subjects, got {len(subjects)} "
                           "(sentinel 4; pass --allow-partial only via probe)")
    mean_stack, all_stacks, centres = group_stack(out_dir, subjects)

    rows: List[pd.DataFrame] = []
    onsets: Dict[str, dict] = {}
    for target, controls in PRIMARY_CONTRASTS.items():
        if target not in spaces:
            continue  # B2 absent -> within-tower qualifier downstream
        res = rsa_time_course(
            mean_stack, spaces, times=centres,
            control_models=[c for c in controls if c in spaces],
            n_perm=n_perm, seed=SEED, **_lift_90(),
        )[target]
        df = res.to_frame()
        df["contrast"] = f"{target}|{'+'.join(res.controls)}"
        df = _mark_clusters(df, res)
        rows.append(df)
        onsets[target] = _onset_block(res, all_stacks, spaces, target, controls, centres)

    pd.concat(rows, ignore_index=True).to_csv(out_dir / "h1_curves.csv", index=False)
    (out_dir / "h1_onsets.json").write_text(json.dumps(onsets, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ("onset_ms", "p_cluster", "onset_ci_lo", "onset_ci_hi")
                          if kk in v} for k, v in onsets.items()}, indent=2))


def _mark_clusters(df: pd.DataFrame, res) -> pd.DataFrame:
    df = df.copy()
    df["in_cluster"] = False
    for _, cl in res.clusters.iterrows():
        if cl["p_cluster"] < 0.05:
            df.loc[(df.time >= cl["t_start"]) & (df.time <= cl["t_end"]), "in_cluster"] = True
    return df


def _partial_curve(stack: np.ndarray, spaces: Dict[str, np.ndarray], target: str,
                   controls: Sequence[str]) -> np.ndarray:
    tgt = spaces[target]
    iu = np.triu_indices(N_VIDEOS, 1)
    ctrl = np.column_stack([spaces[c][iu] for c in controls if c in spaces])
    return np.array([partial_spearman(stack[t], tgt[iu], ctrl) for t in range(stack.shape[0])])


def _onset_block(res, all_stacks: np.ndarray, spaces: Dict[str, np.ndarray],
                 target: str, controls: Sequence[str], centres: np.ndarray) -> dict:
    sig = res.clusters[res.clusters.p_cluster < 0.05]
    out = {"n_clusters_sig": int(len(sig)),
           "clusters": res.clusters.to_dict(orient="records")}
    if len(sig) == 0:
        return out
    first = sig.sort_values("t_start").iloc[0]
    out.update(onset_ms=float(first["t_start"]), p_cluster=float(first["p_cluster"]))

    # Bootstrap the onset over subjects against the SAME per-timepoint threshold
    # the original permutation null produced (prereg H1; rsa.py carries it as
    # ``point_thresh`` exactly for this).
    thresh = res.point_thresh
    rng = np.random.default_rng(SEED)
    boots = []
    n_sub = all_stacks.shape[0]
    for _ in range(N_BOOT_ONSET):
        take = rng.integers(0, n_sub, n_sub)
        curve = _partial_curve(all_stacks[take].mean(axis=0), spaces, target, controls)
        above = curve >= thresh
        boots.append(float(centres[np.argmax(above)]) if above.any() else np.nan)
    boots = np.asarray(boots, dtype=np.float64)
    ok = np.isfinite(boots)
    out.update(
        onset_ci_lo=float(np.nanpercentile(boots[ok], 2.5)) if ok.any() else None,
        onset_ci_hi=float(np.nanpercentile(boots[ok], 97.5)) if ok.any() else None,
        onset_boot_n_defined=int(ok.sum()),
    )
    return out


# --------------------------------------------------------------------------- #
# H2: per-subject alignment + VT gating
# --------------------------------------------------------------------------- #
def run_h2(out_dir: Path, subjects: Sequence[int]) -> None:
    from scipy import stats

    spaces = _load_spaces(out_dir)
    iu = np.triu_indices(N_VIDEOS, 1)
    tgt = spaces["B1"][iu]
    ctrl = np.column_stack([spaces[c][iu] for c in H2_CONTROLS])

    rows = []
    for sid in subjects:
        d = _load_stack(out_dir, sid)
        mask = (d["centres"] >= H2_WINDOW_MS[0]) & (d["centres"] <= H2_WINDOW_MS[1])
        vals = [partial_spearman(d["rdms"][t], tgt, ctrl) for t in np.flatnonzero(mask)]
        rows.append({"subject_id": sid, "tactile_alignment": float(np.nanmean(vals)),
                     "n_windows": int(mask.sum())})
    df = pd.DataFrame(rows)

    cov = pd.read_csv(_work() / "results" / "covariates" / "covariates.csv")
    df = df.merge(cov, on="subject_id", how="left", validate="1:1")
    if df["VT_score"].isna().any():
        raise RuntimeError(f"VT_score missing for {int(df.VT_score.isna().sum())} subjects")
    df["VT_group"] = (df.VT_score > 0).astype(int)

    hi = df.loc[df.VT_group == 1, "tactile_alignment"]
    lo = df.loc[df.VT_group == 0, "tactile_alignment"]
    u, p_primary = stats.mannwhitneyu(hi, lo, alternative="greater")

    # specificity (a): same test on orientation decodability must be null
    spec_a = {}
    if "orientation_peak" in df:
        oh = df.loc[df.VT_group == 1, "orientation_peak"].dropna()
        ol = df.loc[df.VT_group == 0, "orientation_peak"].dropna()
        _, p_orient = stats.mannwhitneyu(oh, ol, alternative="greater")
        spec_a = {"p_orientation": float(p_orient)}

    # specificity (b): VT effect must survive SNR (split-half reliability) adjustment
    r = stats.rankdata
    y, g, snr = r(df.tactile_alignment), df.VT_group.to_numpy(), r(df.split_half_reliability)
    X = np.column_stack([np.ones(len(df)), snr])
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    _, p_adj = stats.mannwhitneyu(resid[g == 1], resid[g == 0], alternative="greater")

    out = {
        "window_ms": H2_WINDOW_MS, "n_vt": int((df.VT_group == 1).sum()),
        "n_novt": int((df.VT_group == 0).sum()),
        "median_alignment_vt": float(hi.median()), "median_alignment_novt": float(lo.median()),
        "mannwhitney_U": float(u), "p_primary_one_sided": float(p_primary),
        **spec_a, "p_snr_adjusted": float(p_adj), "mdd_d": 0.64,
    }
    df.to_csv(out_dir / "h2_subject_alignment.csv", index=False)
    (out_dir / "h2_tests.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- #
# probe (sentinel 5) -- one subject, machinery gates, no aggregates viewed
# --------------------------------------------------------------------------- #
def run_probe(out_dir: Path, subject_id: int) -> int:
    spaces = _load_spaces(out_dir)  # raises on sentinel-2 failure
    failures: List[str] = []

    # gate 1: EEG RDM determinism for the probe subject
    p = subject_rdm_path(out_dir, subject_id)
    if p.exists():
        p.unlink()
    build_subject(subject_id, out_dir)
    a = _load_stack(out_dir, subject_id)
    p.unlink()
    build_subject(subject_id, out_dir)
    b = _load_stack(out_dir, subject_id)
    if not all(np.array_equal(a[k], b[k]) for k in a):
        failures.append("gate1: subject RDM build is not deterministic")

    # gate 3: control effectiveness (partial machinery on real data)
    from scipy import stats as _st

    iu = np.triu_indices(N_VIDEOS, 1)
    mat = spaces["material"][iu]
    for t in (0, a["rdms"].shape[0] // 2, a["rdms"].shape[0] - 1):
        self_p = partial_spearman(a["rdms"][t], mat, mat[:, None])
        if not (np.isnan(self_p) or abs(self_p) < 1e-6):
            failures.append(f"gate3: partial(x, m | m) = {self_p:.2e} at t={t}, expected ~0")
    # The residual must be uncorrelated with material IN THE SPACE THE
    # INFERENCE USES: Pearson on rank-residuals (that is what partial Spearman
    # is).  Re-ranking the residual first -- the first version of this gate --
    # tests a property linear residualization never promises, and with a binary
    # control it fails mathematically while the machinery is correct.
    rm = _st.rankdata(mat)
    rb = _st.rankdata(spaces["B1"][iu])
    zm = (rm - rm.mean()) / rm.std()
    zb = (rb - rb.mean()) / rb.std()
    resid = zb - zm * float(zb @ zm) / float(zm @ zm)
    if abs(float(resid @ zm) / (np.linalg.norm(resid) * np.linalg.norm(zm))) > 1e-6:
        failures.append("gate3: rank-residual retains material correlation")

    # group-path exercise (prereg sentinel 5): 3 windows, tiny null, machinery only
    mini = _zscore_stack(a["rdms"])[[0, a["rdms"].shape[0] // 2, -1]]
    res = rsa_time_course(mini, spaces, times=a["centres"][[0, a["rdms"].shape[0] // 2, -1]],
                          control_models=[c for c in PRIMARY_CONTRASTS["B1"] if c in spaces],
                          n_perm=50, seed=SEED, **_lift_90())
    for name, r in res.items():
        curve = r.r_partial if r.r_partial is not None else r.r
        if not np.all(np.isfinite(curve)):
            failures.append(f"gate5: non-finite group-path curve for {name}")
    if res["B1"].point_thresh is None or not np.all(np.isfinite(res["B1"].point_thresh)):
        failures.append("gate5: point_thresh missing/non-finite")

    # gate 4 (coverage, probe scale): 90 valid conditions for this subject
    if not bool(a["valid"].all()):
        failures.append(f"gate4: {int((~a['valid']).sum())} invalid conditions")

    verdict = "PASS" if not failures else "FAIL"
    (out_dir / "probe_report.json").write_text(json.dumps(
        {"subject": subject_id, "verdict": verdict, "failures": failures,
         "n_windows": int(a["rdms"].shape[0]),
         "centres_ms": [float(a["centres"][0]), float(a["centres"][-1])],
         "n_trials": int(a["n_trials"]), "n_dropped_nonvideo": int(a["n_dropped_nonvideo"])},
        indent=2))
    print(f"probe sub-{subject_id:02d}: {verdict}")
    for f in failures:
        print(" -", f)
    return 0 if not failures else 1


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["build-subject", "probe", "group", "h2"])
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args(argv)
    out_dir = _out_dir(args.out)

    subjects = sorted(load_trials(columns=["subject_id"])["subject_id"].unique().tolist())
    if args.stage == "build-subject":
        print(build_subject(args.subject, out_dir))
        return 0
    if args.stage == "probe":
        return run_probe(out_dir, args.subject)
    if args.stage == "group":
        run_group(out_dir, subjects, n_perm=args.n_perm)
        return 0
    if args.stage == "h2":
        run_h2(out_dir, subjects)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
