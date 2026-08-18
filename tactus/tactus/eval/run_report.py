#!/usr/bin/env python
"""Driver for gate G6: turn finished runs into ``REPORT.md``.

:mod:`tactus.eval.report` owns the *rendering* (and the rule that the design's
statistical resolution is printed before any ablation number); this module is
the missing *driver* that collects what there is to render:

1. per-fold / per-subject retrieval from ``results/runs/<run>/**/aggregate.json``
   and ``per_subject.csv``;
2. the **video-level** permutation null for the primary endpoint, recomputed
   from the saved test embeddings -- plus the trial-level null purely as the
   "how much narrower would the wrong null have been" diagnostic, which is
   never allowed to supply a reported p-value;
3. the split-half **noise ceiling** in the endpoint's own units, so accuracy is
   reported as a fraction of what the data can support rather than of 100%.

Aggregation rule, applied everywhere: average folds *within* an inference unit
first, then bootstrap *across* units.  Bootstrapping over folds or trials is the
same error as trial-level permutation.

CLI
---
    python -m tactus.eval.run_report --run-root $TACTUS_WORK/results/runs \\
        --runs nice_infonce,nice_protonce --regime within_subject \\
        --out $TACTUS_WORK/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .noise_ceiling import fraction_of_ceiling, retrieval_noise_ceiling
from .permutation import (
    null_narrowing_report,
    trial_level_null_diagnostic,
    video_level_permutation_test,
)
from .report import ReportInputs, emit_report

#: decision D4 -- changing this means changing configs/default.yaml and
#: eval/retrieval.py in the same commit.
PRIMARY_KEY = "test/video/g18/top1_pseudo"


def _load_trials_safe() -> Optional[pd.DataFrame]:
    """Trial table for the material strata; ``None`` if unavailable."""
    try:
        from ..common import load_trials
        return load_trials()
    except Exception as exc:  # the report must still render without it
        print(f"[report] material strata unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def find_runs(run_root: Path, names: Optional[Sequence[str]] = None) -> List[Path]:
    out = []
    for p in sorted(run_root.iterdir()):
        if not p.is_dir() or not (p / "aggregate.json").exists():
            continue
        if names and p.name not in names:
            continue
        out.append(p)
    return out


#: chance for one query against a gallery of ``n`` items.
_CHANCE = {"top1": lambda n: 1.0 / n, "top5": lambda n: min(5.0 / n, 1.0),
           "mean_rank": lambda n: (n + 1) / 2.0}


def fold_dirs(run: Path, regime: Optional[str]) -> List[Path]:
    """Fold directories of ``run`` belonging to ``regime``.

    A run directory is keyed on ``run.name``, NOT on the regime, so training the
    same config under two regimes writes both into ``results/runs/<name>/folds/``
    (within_subject folds are ``vfNN``, double_disjoint ``vfNN_sfNN``).  A report
    that globs the directory then averages 5 within-subject folds together with
    32 double-disjoint folds and silently reports a number belonging to neither.
    Observed live: nice_protonce read 11.93% (within_subject, 5 folds) before the
    double-disjoint grid started writing and 11.49% after.

    Each fold stamps its own ``metrics.json["regime"]``, so filter on that and
    refuse to guess when a directory turns out to be mixed.
    """
    out: List[Path] = []
    seen: Dict[str, int] = {}
    for m in sorted((run / "folds").glob("*/metrics.json")) if (run / "folds").is_dir() else []:
        try:
            reg = str(json.loads(m.read_text()).get("regime", "unknown"))
        except Exception:
            reg = "unknown"
        seen[reg] = seen.get(reg, 0) + 1
        if regime is None or reg == regime:
            out.append(m.parent)
    if len(seen) > 1:
        print(f"[report] {run.name}: folds span {seen}; keeping regime={regime!r} "
              f"({len(out)} fold(s))", file=sys.stderr)
    return out


def collect_retrieval(runs: Sequence[Path], regime: Optional[str] = None) -> pd.DataFrame:
    """Per-subject retrieval across runs, in the schema ``aggregate_folds`` wants.

    ``report.aggregate_folds`` groups on ``(direction, trial_type, gallery,
    metric)`` and bootstraps ``value`` across ``subject_id``.  The trainer writes
    one wide row per (subject, unit, trial_type), so melt it.  The inference unit
    is the **subject** -- these are fixed-stimulus claims (blueprint 5.2).
    """
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for fd in fold_dirs(run, regime):
            csv = fd / "per_subject.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            fold = fd.name
            for _, r in df.iterrows():
                unit = str(r.get("unit", "video"))
                n_items = 18 if unit == "video" else 72
                # Vocabulary must match tactus.eval.retrieval.primary_endpoint,
                # or section 2 of the report silently reports "NOT RUN":
                #   trial_type "pseudo4" (not "pseudo_k4"), gallery "nway18".
                ttype = str(r.get("trial_type", "single"))
                if ttype.startswith("pseudo"):
                    ttype = "pseudo" + "".join(ch for ch in ttype if ch.isdigit())
                for metric in ("top1", "top5", "mean_rank"):
                    if metric not in df.columns or pd.isna(r.get(metric)):
                        continue
                    rows.append({
                        "run": run.name,
                        "video_fold_id": fold,
                        "subject_id": int(r.get("subject_id", -1)),
                        "direction": "eeg2vid",
                        "trial_type": ttype,
                        "gallery": f"nway{n_items}",
                        "metric": metric,
                        "value": float(r[metric]),
                        "chance": _CHANCE[metric](n_items),
                        "n": r.get("n", np.nan),
                    })
    return pd.DataFrame(rows)


def collect_primary(runs: Sequence[Path], regime: Optional[str] = None) -> pd.DataFrame:
    """One row per (run, fold) carrying the pre-registered primary endpoint.

    Never falls back to ``aggregate.json``: that file is written per RUN, not per
    regime, so on a mixed directory it is the very average this filter exists to
    avoid.
    """
    rows = []
    for run in runs:
        for fd in fold_dirs(run, regime):
            f = fd / "test_metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            if PRIMARY_KEY in m:
                rows.append({"run": run.name, "fold": fd.name,
                             "primary": float(m[PRIMARY_KEY])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the two nulls
# --------------------------------------------------------------------------- #
def _pseudo_trials(z: np.ndarray, cond: np.ndarray, subj: np.ndarray, k: int,
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Average k repeats within (subject, condition); returns (Z, condition_id)."""
    key = subj.astype(np.int64) * 100000 + cond.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    out_z, out_c = [], []
    for grp in np.split(np.arange(len(ks)), bounds):
        if len(grp) < k:
            continue
        idx = rng.permutation(grp)
        for j in range(len(idx) // k):
            out_z.append(zs[idx[j * k:(j + 1) * k]].mean(0))
            out_c.append(ks[grp[0]] % 100000)
    if not out_z:
        return np.zeros((0, z.shape[1])), np.zeros(0, dtype=np.int64)
    Z = np.stack(out_z)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return Z, np.asarray(out_c, dtype=np.int64)


def _material_strata(gal_ids: np.ndarray, trials: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    """Material label per gallery video, for the material-matched null."""
    if trials is None or "material" not in trials.columns:
        return None
    m = (trials.drop_duplicates("video_id").set_index("video_id")["material"].astype(str))
    try:
        return np.asarray([m.loc[int(v)] for v in gal_ids])
    except KeyError:
        return None


def primary_permutation(
    emb_paths: Sequence[Path], *, n_perm: int = 1000, k: int = 4, seed: int = 0,
    trials: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Video-level permutation null for the primary endpoint, pooled over folds.

    The statistic is exactly the endpoint: eeg->video top-1 in the held-out
    18-video gallery, on k=4 pseudo-trials, averaged over folds.  A permutation
    relabels the **gallery videos** -- the exchangeable unit -- so the null keeps
    the clustered structure of the 32 trials that share a source video.
    """
    rng = np.random.default_rng(seed)
    folds = []
    for p in emb_paths:
        d = np.load(p, allow_pickle=True)
        z = d["z_eeg"].astype(np.float64)
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        gal_ids = d["gallery_video_ids"].astype(np.int64)
        gal = d["z_vid_video"].astype(np.float64)
        gal /= np.linalg.norm(gal, axis=1, keepdims=True) + 1e-12
        Z, cond = _pseudo_trials(z, d["condition_id"].astype(np.int64),
                                 d["subject_id"].astype(np.int64), k, rng)
        if not len(Z):
            continue
        vid = cond // 4 + 1                       # condition_id -> video_id
        pos = {int(v): i for i, v in enumerate(gal_ids)}
        tgt = np.array([pos.get(int(v), -1) for v in vid])
        keep = tgt >= 0
        folds.append((Z[keep] @ gal.T, tgt[keep], len(gal_ids)))
        gal_ids_ref = gal_ids

    if not folds:
        raise RuntimeError("no usable test_embeddings.npz")

    def stat(perm_base: np.ndarray) -> float:
        """perm_base indexes the gallery; identity gives the observed value."""
        accs = []
        for sims, tgt, n_gal in folds:
            p = perm_base[:n_gal] % n_gal
            accs.append(float(np.mean(sims.argmax(1) == p[tgt])))
        return float(np.mean(accs))

    n_gal = folds[0][2]
    video = video_level_permutation_test(
        stat, n_base=n_gal, n_perm=n_perm, seed=seed,
        statistic_name=PRIMARY_KEY,
    )

    # Material-matched null: shuffle only WITHIN material, so a model that has
    # learned nothing but the 8-way material code scores at the null.  The gap
    # between this and the plain null is the share of the endpoint that material
    # identity alone explains -- the attribute-shortcut quantification the
    # blueprint (5.2) asks for, rather than the inflated cross-group figure.
    strata = _material_strata(np.asarray(gal_ids_ref), trials)
    matched = None
    n_singleton = 0
    if strata is not None and len(np.unique(strata)) > 1:
        # A gallery video that is the ONLY member of its material is a fixed
        # point of a within-material permutation: the null preserves its exact
        # identity, not merely its material.  Those queries therefore score at
        # the model's real accuracy inside the null, which inflates the null for
        # a reason that has nothing to do with material knowledge and makes the
        # surviving fraction look smaller than it is.  Measured here: 26/216
        # gallery slots (12%) are such singletons.  Score the matched null only
        # on queries whose target has at least one same-material companion.
        uniq, counts = np.unique(strata, return_counts=True)
        size_of = dict(zip(uniq.tolist(), counts.tolist()))
        keep_gal = np.array([size_of[s] > 1 for s in strata])
        n_singleton = int((~keep_gal).sum())

        def stat_matched(perm_base: np.ndarray) -> float:
            accs = []
            for sims, tgt, n_g in folds:
                pm = perm_base[:n_g] % n_g
                sel = keep_gal[tgt]                    # drop singleton-target queries
                if not sel.any():
                    continue
                accs.append(float(np.mean(sims[sel].argmax(1) == pm[tgt[sel]])))
            return float(np.mean(accs)) if accs else float("nan")

        matched = video_level_permutation_test(
            stat_matched, n_base=n_gal, n_perm=n_perm, seed=seed, strata=strata,
            statistic_name=PRIMARY_KEY + " (material-matched, non-singleton targets)",
        )

    # The deliberately WRONG null, for the narrowing table only.
    n_tr = int(sum(len(t) for _, t, _ in folds))

    def stat_trial(perm_trials: np.ndarray) -> float:
        accs, off = [], 0
        for sims, tgt, _ in folds:
            take = perm_trials[off:off + len(tgt)] % len(tgt)
            off += len(tgt)
            accs.append(float(np.mean(sims.argmax(1) == tgt[take])))
        return float(np.mean(accs))

    trial = trial_level_null_diagnostic(
        stat_trial, n_tr, n_perm=n_perm, seed=seed, observed=video.observed,
        statistic_name=PRIMARY_KEY,
    )
    narrowing = null_narrowing_report(video, trial)

    def _summary(res) -> Dict[str, Any]:
        """Serialisable summary WITHOUT the raw null array.

        The array is thousands of floats; rendered into a markdown cell it
        drowns the table it is supposed to support.  Everything a reader needs
        (mean, sd, quantiles, z, p) is already summarised.
        """
        d = res.to_dict() if hasattr(res, "to_dict") else dict(vars(res))
        d.pop("null", None)
        return d

    out: Dict[str, Any] = {
        "video_level": _summary(video),
        "trial_level": _summary(trial),
        "narrowing": narrowing,
    }
    if matched is not None:
        out["video_level_material_matched"] = _summary(matched)
        denom = matched.observed - video.null_mean
        out["material_matched"] = {
            "observed": matched.observed,
            "observed_all_targets": video.observed,
            "n_singleton_material_gallery_slots": n_singleton,
            "plain_null_mean": video.null_mean,
            "material_matched_null_mean": matched.null_mean,
            "p_material_matched": matched.p_value,
            "frac_of_effect_beyond_material": (
                round(float((matched.observed - matched.null_mean) / denom), 4)
                if abs(denom) > 1e-12 else None
            ),
            "note": "fraction of the above-chance endpoint that survives a null which "
                    "preserves material identity; the remainder is explained by the "
                    "8-way material code alone. Queries whose target is the only "
                    "gallery video of its material are excluded from BOTH terms: a "
                    "within-material permutation is the identity on them, so they "
                    "would inflate the null for a non-material reason.",
        }
    return out


def primary_ceiling(emb_paths: Sequence[Path], *, k: int = 4, seed: int = 0,
                    n_resamples: int = 20,
                    n_gallery_subjects: Optional[int] = 10,
                    n_gallery_draws: int = 20) -> pd.DataFrame:
    """Split-half EEG->EEG ceiling in the endpoint's units, pooled over folds."""
    frames = []
    for p in emb_paths:
        d = np.load(p, allow_pickle=True)
        try:
            df = retrieval_noise_ceiling(
                d["z_eeg"], d["video_id"], d["subject_id"], k=k,
                n_resamples=n_resamples, gallery_sizes=(2, 10, 18), seed=seed,
                per_subject=True,
                # One common denominator across regimes (DECISIONS D15): a
                # within_subject fold has 80 test subjects and a double_disjoint
                # fold has 10, and the pooled gallery is cleaner the more
                # subjects it averages, so an unpinned ceiling is a property of
                # the fold design rather than of the data.
                n_gallery_subjects=n_gallery_subjects,
                n_gallery_draws=n_gallery_draws,
            )
        except Exception as exc:  # keep the report honest rather than crashing
            print(f"[ceiling] {p.parent.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        df["fold"] = p.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tactus.eval.run_report")
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--runs", default=None, help="comma-separated run names")
    ap.add_argument("--regime", default="within_subject")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--pseudo-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-permutation", action="store_true")
    ap.add_argument("--skip-ceiling", action="store_true")
    ap.add_argument("--ceiling-subjects", type=int, default=10,
                    help="subjects averaged into the split-half gallery. Pinned so the "
                         "denominator is identical across regimes (D15); 10 is the "
                         "double-disjoint fold size, the largest value both regimes have.")
    ap.add_argument("--ceiling-draws", type=int, default=20,
                    help="independent subject subsets averaged into the pooled "
                         "ceiling. Pinning only the count is not enough: which 10 "
                         "subjects are drawn moves the ceiling from 0.1122 to 0.1539 "
                         "across eight seeds on one fold, more than the accuracy gaps "
                         "being compared (DECISIONS D15).")
    args = ap.parse_args(argv)

    names = [s.strip() for s in args.runs.split(",")] if args.runs else None
    runs = find_runs(args.run_root, names)
    if not runs:
        print(f"no finished runs under {args.run_root}", file=sys.stderr)
        return 2
    print(f"[report] runs: {[r.name for r in runs]}")

    retrieval = collect_retrieval(runs, args.regime)
    primary = collect_primary(runs, args.regime)
    notes: List[str] = []
    uncertifiable = [
        "Any claim that the alignment in windows beyond ~150 ms is not eye-movement "
        "driven. The dataset ships 0 EOG channels (verified in the BDF header: 64 EEG "
        "+ Status only), so the ocular controls are frontal surrogates plus ICA, and "
        "a surrogate-beating model does not exclude saccadic spike potentials volume-"
        "conducting to posterior sites.",
        "Attribute-specific claims separating toucher / object / material. Audit C "
        "measured these at |association| 0.99-1.00 across the 90 stimuli: they are "
        "one variable here, not three.",
        "Any anatomical localisation (TPJ/pSTS vs S1) from 64-channel sensor-space EEG.",
    ]

    perm = None
    if not args.skip_permutation:
        emb = [d / "test_embeddings.npz" for d in fold_dirs(runs[0], args.regime)]
        emb = sorted(p for p in emb if p.exists())
        if emb:
            print(f"[report] video-level permutation over {len(emb)} fold(s), "
                  f"n_perm={args.n_perm} ...")
            perm = primary_permutation(emb, n_perm=args.n_perm, k=args.pseudo_k,
                                       seed=args.seed, trials=_load_trials_safe())
            notes.append(
                f"Primary-endpoint permutation used {len(emb)} fold(s) of "
                f"{runs[0].name}; the exchangeable unit is the source video."
            )

    ceiling = None
    if not args.skip_ceiling:
        emb = [d / "test_embeddings.npz" for d in fold_dirs(runs[0], args.regime)]
        emb = sorted(p for p in emb if p.exists())
        if emb:
            print(f"[report] split-half noise ceiling over {len(emb)} fold(s) ...")
            ceiling = primary_ceiling(emb, k=args.pseudo_k, seed=args.seed,
                                      n_gallery_subjects=args.ceiling_subjects,
                                      n_gallery_draws=args.ceiling_draws)
            if ceiling is not None and len(ceiling):
                obs = primary["primary"].mean() if len(primary) else np.nan
                # retrieval_noise_ceiling returns long form:
                #   subject_id | endpoint | ceiling | ...
                # the endpoint matching D4 is the 18-way top-1, pooled subjects.
                sub = ceiling[(ceiling["endpoint"] == "nway18_top1")
                              & (ceiling["subject_id"].astype(str) == "pooled")]
                if len(sub):
                    ceil_val = float(np.nanmean(sub["ceiling"].to_numpy(dtype=float)))
                    # The denominator has its own sampling noise -- which subjects
                    # land in the pooled gallery moves it by more than the gaps
                    # between arms.  Quote the band, or the fraction reads as if
                    # it were exact (DECISIONS D15).
                    sd = float(np.nanmean(sub.get("ceiling_sd", pd.Series([np.nan]))
                                          .to_numpy(dtype=float))) \
                        if "ceiling_sd" in sub else float("nan")
                    band = ""
                    if np.isfinite(sd) and sd > 0 and ceil_val > 0:
                        lo = fraction_of_ceiling(obs, ceil_val + sd)
                        hi = fraction_of_ceiling(obs, max(ceil_val - sd, 1e-9))
                        band = (f" Denominator +-1 sd moves this to "
                                f"[{lo:.3f}, {hi:.3f}] (ceiling sd {sd:.4f} over "
                                f"{int(sub.get('n_gallery_draws', pd.Series([0])).max())} "
                                f"subject draws), so differences narrower than that "
                                f"band are not resolvable.")
                    notes.append(
                        f"fraction-of-ceiling for {PRIMARY_KEY}: "
                        f"{fraction_of_ceiling(obs, ceil_val):.3f} "
                        f"(observed {obs:.4f}, split-half ceiling {ceil_val:.4f}).{band} "
                        "The ceiling is reliability-matched, not a hard bound: the "
                        "video gallery is noiseless, so exceeding it is informative "
                        "rather than paradoxical."
                    )

    if len(primary):
        for run, grp in primary.groupby("run"):
            notes.append(f"{run}: {PRIMARY_KEY} = {grp['primary'].mean():.4f} "
                         f"over {len(grp)} fold(s) (chance 0.0556)")

    args.out.mkdir(parents=True, exist_ok=True)
    inputs = ReportInputs(
        run_name=" + ".join(r.name for r in runs),
        regime=args.regime,
        retrieval_long=retrieval if len(retrieval) else None,
        noise_ceiling_long=ceiling if ceiling is not None and len(ceiling) else None,
        permutation=perm,
        notes=notes,
        uncertifiable_claims=uncertifiable,
    )
    path = emit_report(inputs, args.out)
    print(f"[written] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
