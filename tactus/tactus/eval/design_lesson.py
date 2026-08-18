"""Stimulus-set structure that constrains what any analysis here can claim (D17).

Three properties of the 90 videos, none of them an analysis choice, all of them
load-bearing for how results on this dataset must be worded:

1. **Attribute collinearity.** ``toucher``, ``object`` and ``material`` are the
   same variable on this stimulus set. A claim that separates them is not weakly
   supported, it is unidentifiable.
2. **Empty cells.** The material x touch_type table is mostly empty, so the two
   factors cannot be crossed.
3. **Class imbalance.** Every categorical attribute except orientation is
   strongly imbalanced, which is what makes accuracy-against-uniform-chance
   read as decoding when a majority-class predictor would score the same (D19).

This is a property of the stimuli, which were built to sample naturalistic touch
events rather than to cross their attributes orthogonally. Nothing here is a
criticism of the dataset or of the companion paper's analyses; it is the
constraint any user of the dataset inherits.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

#: Attributes whose pairwise separation this stimulus set cannot support.  Kept
#: hard-coded (D17a) so a future analysis has to delete a line to make the claim.
UNANSWERABLE = (("toucher", "material"), ("toucher", "object"), ("object", "material"))

#: Attributes of the base video.  ``orientation`` is deliberately absent: it is a
#: property of the *condition*, not of the video, so dropping duplicates on
#: video_id would sample whichever orientation happened to come first and report
#: a majority rate of 0.278 for a factor that is exactly balanced by design.
ATTRS = ["material", "toucher", "object", "touch_type", "approaching"]
CONTINUOUS = ["valence", "arousal", "threat", "pain"]


def cramers_v(a: pd.Series, b: pd.Series, corrected: bool = True) -> float:
    """Cramer's V between two categorical columns.

    ``corrected`` applies the Bergsma-Wicher bias correction, which matters here:
    the material x touch_type table is 61/96 empty and the uncorrected statistic
    is inflated on sparse tables.  Both are reported, because the Phase-0 audit
    quoted the uncorrected values (toucher/material 1.000, object/material 0.993)
    and the two must be reconcilable rather than silently different.  The
    correction can drive V to exactly 0 when phi^2 falls below the expected
    chance inflation; that is the estimator saying the table is too sparse to
    support a claim, not evidence of independence.
    """
    tab = pd.crosstab(a, b).to_numpy(dtype=np.float64)
    n = tab.sum()
    if n == 0 or min(tab.shape) < 2:
        return float("nan")
    chi2 = 0.0
    exp = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((tab - exp) ** 2 / np.where(exp > 0, exp, np.nan)))
    phi2 = chi2 / n
    r, k = tab.shape
    # Bergsma-Wicher correction: without it V is inflated on sparse tables, and
    # this table is 61/96 empty.
    if not corrected:
        denom = min(k - 1, r - 1)
        return float(np.sqrt(phi2 / denom)) if denom > 0 else float("nan")
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def build(trials: pd.DataFrame) -> dict:
    """All three tables, computed over the 90 base videos (not over trials).

    The video is the unit: counting trials would multiply every cell by 8 x 80
    and make sparsity look like precision.
    """
    vids = trials.drop_duplicates("video_id").sort_values("video_id")
    cols = [c for c in ATTRS if c in vids.columns]

    coll = pd.DataFrame(index=cols, columns=cols, dtype=float)
    raw = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a, b in itertools.combinations_with_replacement(cols, 2):
        v = 1.0 if a == b else cramers_v(vids[a], vids[b])
        u = 1.0 if a == b else cramers_v(vids[a], vids[b], corrected=False)
        coll.loc[a, b] = coll.loc[b, a] = v
        raw.loc[a, b] = raw.loc[b, a] = u

    cont = [c for c in CONTINUOUS if c in vids.columns]
    affect = vids[cont].corr(method="pearson") if cont else pd.DataFrame()

    cross = pd.crosstab(vids["material"], vids["touch_type"])
    n_cells = int(cross.shape[0] * cross.shape[1])
    n_empty = int((cross.to_numpy() == 0).sum())

    imb = []
    # Orientation is crossed with every video by construction (90 videos x 4
    # orientations = the 360 conditions), so it is scored over conditions, where
    # it is exactly uniform and exactly independent of every video attribute.
    conds = trials.drop_duplicates("condition_id")
    n_o = int(conds["orientation"].nunique())
    o_rate = float(conds["orientation"].value_counts(normalize=True).max())
    imb.append({"attribute": "orientation (over conditions)", "n_classes": n_o,
                "uniform_chance": 1.0 / n_o, "majority_class": "-- balanced --",
                "majority_rate": o_rate, "majority_over_uniform": o_rate * n_o})
    for c in cols:
        vc = vids[c].value_counts(normalize=True).sort_values(ascending=False)
        imb.append({"attribute": c, "n_classes": int(vids[c].nunique()),
                    "uniform_chance": 1.0 / vids[c].nunique(),
                    "majority_class": str(vc.index[0]), "majority_rate": float(vc.iloc[0]),
                    "majority_over_uniform": float(vc.iloc[0] * vids[c].nunique())})
    return {"collinearity": coll, "collinearity_uncorrected": raw,
            "affect": affect, "cross": cross,
            "n_cells": n_cells, "n_empty": n_empty,
            "imbalance": pd.DataFrame(imb).sort_values("majority_over_uniform",
                                                       ascending=False),
            "n_videos": int(len(vids))}


def render(d: dict) -> str:
    coll = d["collinearity"]
    pairs = [(a, b, float(coll.loc[a, b]))
             for a, b in itertools.combinations(coll.index, 2)]
    high = sorted([p for p in pairs if p[2] > 0.5], key=lambda p: -p[2])
    unans = ", ".join(f"`{a}`/`{b}` = {float(coll.loc[a, b]):.3f}"
                      for a, b in UNANSWERABLE if a in coll.index and b in coll.index)
    return (
        "# Design lesson: what this stimulus set can and cannot separate\n\n"
        f"Computed over the {d['n_videos']} base videos. The video is the unit -- "
        "counting trials would multiply every cell by 8 repeats x 80 subjects and "
        "make sparsity look like precision.\n\n"
        "## 1. Attribute collinearity (bias-corrected Cramer's V)\n\n"
        + coll.round(3).to_markdown() +
        "\n\nUncorrected, for reconciliation with the Phase-0 audit, which "
        "quoted these:\n\n" + d["collinearity_uncorrected"].round(3).to_markdown() +
        f"\n\n{len(high)} of {len(pairs)} pairs exceed 0.5. The three the "
        f"analysis treats as unanswerable: {unans}.\n\n"
        "On this stimulus set a hand touches skin and an object touches everything "
        "else, so **\"material decoding\" and \"hand-versus-object decoding\" are two "
        "names for one claim.** Any result reported for one of them is the same "
        "result reported for the other, and neither can be presented as converging "
        "evidence for the other. This is a property of a stimulus set built to "
        "sample naturalistic touch events rather than to cross its attributes, not "
        "an error in any analysis of it.\n\n"
        "## 2. Affect axes (Pearson r over videos)\n\n"
        + (d["affect"].round(3).to_markdown() if len(d["affect"]) else "(unavailable)") +
        "\n\n## 3. material x touch_type occupancy\n\n"
        + d["cross"].to_markdown() +
        f"\n\n**{d['n_empty']} of {d['n_cells']} cells are empty.** The two factors "
        "cannot be crossed, so no interaction between them is estimable and any "
        "main effect of one is read through the marginal distribution of the other.\n\n"
        "## 4. Class imbalance\n\n"
        + d["imbalance"].round(3).to_markdown(index=False) +
        "\n\nRead `majority_over_uniform` as the apparent decoding factor a "
        "majority-class predictor earns when accuracy is scored against uniform "
        "chance. Orientation is the only attribute where it is exactly 1.0 -- it is "
        "crossed with video by construction rather than sampled -- and it is the "
        "only attribute whose MVPA replicates (D19). Quote the majority rate beside "
        "every accuracy computed on this dataset.\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tactus.eval.design_lesson")
    ap.add_argument("--trials", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    if a.trials is not None:
        trials = pd.read_parquet(a.trials)
    else:
        from tactus.common import load_trials
        trials = load_trials()
    d = build(trials)
    a.out.mkdir(parents=True, exist_ok=True)
    d["collinearity"].to_csv(a.out / "collinearity.csv")
    d["collinearity_uncorrected"].to_csv(a.out / "collinearity_uncorrected.csv")
    d["cross"].to_csv(a.out / "material_x_touchtype.csv")
    d["imbalance"].to_csv(a.out / "class_imbalance.csv", index=False)
    text = render(d)
    (a.out / "DESIGN_LESSON.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
