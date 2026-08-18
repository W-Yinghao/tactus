"""Per-subject covariate table for the phenotype analysis (DECISIONS D13, Q3).

Q3 asks whether individual differences in the EEG-video alignment track the
questionnaire scores ds005662 ships (VT, EQ, IRI, MTS).  Any such claim lives or
dies on the nuisance covariates, because "this subject aligns better" and "this
subject had cleaner data" are the same sentence unless the second is regressed
out.  This module assembles all of them into one table from artefacts that
already exist, so the analysis cannot quietly use a different set than it
reports.

D13's ordering is baked in:

* primary SNR covariate = **per-subject split-half reliability**;
* secondary = the repaired scale-invariant ISC ratio;
* the pre-repair ISC column is not exported at all -- it correlated rho = 0.19
  with reliability, so it was not measuring SNR, and voiding it means removing
  it rather than annotating it.

The behavioural covariate needs a correction to how the task is usually
described.  There is no per-target hit rate in this dataset: ``rt``/``resp`` are
populated on 32 rows per subject, none of them target rows, and they are not
adjacent to targets.  The task is to **count the targets within each of the 32
sequences and report the count at the end**; ``cresp`` is the true count and
``resp`` the reported one.  So the attention measure is per-sequence counting
accuracy, which has usable spread (mean 0.81, sd 0.19, 9/80 at ceiling) --
unlike a hit rate, which does not exist here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

#: Questionnaire columns in participants.tsv.  These are Q3's *outcomes*, not
#: covariates, and are carried here only so one file holds the whole design.
PHENOTYPES = ["VT_score", "EQ_score", "IRI_score", "MTS"]

#: Exported from the CorrCA per-subject table.  The pre-repair ISC column is
#: deliberately absent -- see the module docstring.
CORRCA_COLS = ["split_half_reliability", "split_half_reliability_stimulus_specific",
               "isc_ratio_sum", "amplitude_rms", "channel_sd_ratio"]


def behavioural(bids_root: Path) -> pd.DataFrame:
    """Per-sequence counting accuracy and report latency, per subject."""
    rows = []
    for p in sorted(glob.glob(str(bids_root / "sub-*" / "eeg" / "*_events.tsv"))):
        sid = int(os.path.basename(p).split("-")[1][:2])
        d = pd.read_csv(p, sep="\t")
        d["_resp"] = pd.to_numeric(d.get("resp"), errors="coerce")
        d["_rt"] = pd.to_numeric(d.get("rt"), errors="coerce")
        # One response per sequence, logged on whichever row it landed on.
        g = d.groupby("sequencenumber").agg(n_target=("istarget", "sum"),
                                            resp=("_resp", "max"), rt=("_rt", "max"))
        rows.append({
            "subject_id": sid,
            "n_sequences": int(len(g)),
            "count_accuracy": float((g.n_target == g.resp).mean()),
            "count_abs_error": float((g.resp - g.n_target).abs().mean()),
            "report_rt_mean": float(g.rt.mean()), "report_rt_sd": float(g.rt.std()),
        })
    return pd.DataFrame(rows)


def data_quality(epoch_dir: Path, window: str) -> pd.DataFrame:
    """Retained-trial count and artefact fraction, from the epoch sidecars."""
    rows = []
    for p in sorted(glob.glob(str(epoch_dir / f"sub-*_{window}.json"))):
        j = json.loads(Path(p).read_text())
        ei = j.get("epoch_info", {})
        rows.append({
            "subject_id": int(j["subject_id"]),
            "n_trials_kept": int(ei.get("n_kept", 0)),
            "n_trials_dropped": int(j.get("n_dropped", 0)),
            "frac_abs_gt_20": float(ei.get("frac_abs_gt_20", np.nan)),
            "abs_max_after_scaling": float(ei.get("abs_max_after_scaling", np.nan)),
            "n_bad_channels_interpolated": len(
                j.get("raw_meta", {}).get("bad_channels_interpolated", [])),
        })
    return pd.DataFrame(rows)


def orientation_decodability(curves_npz: Path) -> pd.DataFrame:
    """Peak orientation decoding per subject -- D13's non-social control axis.

    A subject whose orientation decoding is high has a usable evoked response;
    if a phenotype effect survives every other covariate but not this one, it was
    signal quality wearing a questionnaire's name.
    """
    d = np.load(curves_npz)
    curves, chance = np.asarray(d["curves"]), float(d["chance"])
    return pd.DataFrame({
        "subject_id": np.asarray(d["subjects"]).astype(int),
        "orientation_peak": curves.max(axis=1),
        "orientation_peak_above_chance": curves.max(axis=1) - chance,
        "orientation_mean": curves.mean(axis=1),
    })


def build(bids_root: Path, epoch_dir: Path, window: str, corrca_csv: Path,
          curves_npz: Optional[Path]) -> pd.DataFrame:
    part = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    part["subject_id"] = part["participant_id"].str.split("-").str[1].astype(int)
    keep = ["subject_id", "sex", "age", "handedness"] + \
           [c for c in PHENOTYPES if c in part.columns]
    df = part[keep]

    cor = pd.read_csv(corrca_csv)
    df = df.merge(cor[["subject_id"] + [c for c in CORRCA_COLS if c in cor.columns]],
                  on="subject_id", how="left")
    df = df.merge(behavioural(bids_root), on="subject_id", how="left")
    df = df.merge(data_quality(epoch_dir, window), on="subject_id", how="left")
    if curves_npz is not None and Path(curves_npz).exists():
        df = df.merge(orientation_decodability(Path(curves_npz)),
                      on="subject_id", how="left")
    return df.sort_values("subject_id").reset_index(drop=True)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Spread and pairwise collinearity -- a covariate with no variance is not one.

    Reported before any modelling, because the useful failure mode here is a
    covariate that cannot do its job (everyone at ceiling) rather than one that
    is missing.
    """
    num = df.select_dtypes(include=[np.number]).drop(columns=["subject_id"],
                                                     errors="ignore")
    out = pd.DataFrame({
        "n_present": num.notna().sum(),
        "mean": num.mean(), "sd": num.std(),
        "min": num.min(), "max": num.max(),
        "n_unique": num.nunique(),
        "frac_at_mode": num.apply(
            lambda c: float(c.value_counts(normalize=True).iloc[0]) if c.notna().any()
            else np.nan),
    })
    out["usable"] = (out["sd"] > 0) & (out["n_unique"] > 3) & (out["frac_at_mode"] < 0.9)
    return out.round(4)


def power_block(df: pd.DataFrame) -> pd.DataFrame:
    """What effect size each Q3 outcome can actually resolve at n = 80.

    Written before the analysis rather than after, because two of the four
    outcomes are far weaker than "n = 80" suggests: VT_score is zero for 47 of
    80 subjects and MTS splits 17/63.  A design that cannot resolve the effect
    is not evidence of absence, and saying so afterwards is worth much less.
    """
    from scipy import stats

    def mdd_r(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
        # Fisher z: z_r = (z_a/2 + z_b) / sqrt(n - 3)
        z = (stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power)) / np.sqrt(n - 3)
        return float(np.tanh(z))

    def mdd_d(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
        return float((stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power))
                     * np.sqrt(1.0 / n1 + 1.0 / n2))

    rows = []
    n = len(df)
    for c in ("EQ_score", "IRI_score"):
        if c in df:
            rows.append({"outcome": c, "test": "Spearman over subjects",
                         "n_effective": n, "mdd": mdd_r(n), "mdd_kind": "correlation r",
                         "note": "continuous, well spread"})
    if "VT_score" in df:
        nz = int((df.VT_score > 0).sum())
        rows.append({"outcome": "VT_score (as continuous)", "test": "Spearman over subjects",
                     "n_effective": n, "mdd": mdd_r(n), "mdd_kind": "correlation r",
                     "note": f"OPTIMISTIC: {n - nz}/{n} subjects are exactly 0, so the "
                             "rank test is mostly one big tie"})
        rows.append({"outcome": "VT_score (binarised >0)", "test": "two-sample",
                     "n_effective": min(nz, n - nz), "mdd": mdd_d(nz, n - nz),
                     "mdd_kind": "Cohen's d", "note": f"{nz} vs {n - nz}"})
    if "MTS" in df:
        y = int((df.MTS.astype(str).str.lower() == "yes").sum())
        rows.append({"outcome": "MTS", "test": "two-sample", "n_effective": min(y, n - y),
                     "mdd": mdd_d(y, n - y), "mdd_kind": "Cohen's d",
                     "note": f"{y} yes vs {n - y} no"})
    return pd.DataFrame(rows).round(3)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tactus.eval.covariates")
    ap.add_argument("--bids-root", required=True, type=Path)
    ap.add_argument("--epoch-dir", required=True, type=Path)
    ap.add_argument("--window", default="w0600")
    ap.add_argument("--corrca-csv", required=True, type=Path,
                    help="per_subject_isc.csv from tactus.baselines.corrca --split-half")
    ap.add_argument("--orientation-curves", type=Path, default=None,
                    help="curves.npz from the orientation MVPA")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    df = build(a.bids_root, a.epoch_dir, a.window, a.corrca_csv, a.orientation_curves)
    aud = audit(df)
    pw = power_block(df)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "covariates.csv", index=False)
    aud.to_csv(a.out / "covariate_audit.csv")
    pw.to_csv(a.out / "q3_power.csv", index=False)

    num = df.select_dtypes(include=[np.number]).drop(columns=["subject_id"],
                                                     errors="ignore")
    corr = num.corr(method="spearman").round(3)
    corr.to_csv(a.out / "covariate_correlations.csv")
    unusable = aud.index[~aud["usable"].astype(bool)].tolist()

    (a.out / "COVARIATES.md").write_text(
        "# Q3 covariate table (DECISIONS D13)\n\n"
        f"- {len(df)} subjects, window `{a.window}`\n"
        "- primary SNR covariate: `split_half_reliability`; secondary: "
        "`isc_ratio_sum`. The pre-repair ISC column is not exported -- it "
        "correlated rho = 0.19 with reliability, so voiding it means removing "
        "it, not annotating it.\n"
        "- `count_accuracy` is per-sequence target *counting*, not a hit rate. "
        "This dataset has no per-target detection: the 32 responses per subject "
        "are one report per sequence, and none of them sit on a target row.\n\n"
        "## What Q3 can resolve\n\n" + pw.to_markdown(index=False) +
        "\n\nRead this before the result, not after. Two of the four outcomes are "
        "much weaker than n = 80 implies: VT_score is exactly 0 for 47 of 80 "
        "subjects and MTS splits 17/63, so both are effectively small two-group "
        "comparisons. A null on either is a statement about the design.\n\n"
        "## Spread\n\n" + aud.to_markdown() +
        ("\n\n**Not usable as covariates** (no spread, or >90% of subjects at one "
         "value): " + ", ".join(f"`{c}`" for c in unusable) + "\n" if unusable else
         "\n\nEvery numeric column has usable spread.\n") +
        "\n## Spearman correlations\n\n" + corr.to_markdown() + "\n")
    print(pw.to_string(index=False))
    print()
    print(aud.to_string())
    if unusable:
        print("\nnot usable:", unusable)
    print(f"\n[written] {a.out / 'COVARIATES.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
