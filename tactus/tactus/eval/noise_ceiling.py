"""Split-half noise ceilings from the 8 repeats per condition.

BLUEPRINT v2 §5.2: "noise ceilings everywhere".  Three uses:

(a) a per-subject *attainable* retrieval ceiling -- what any model could reach
    given how reliable that subject's evoked responses actually are;
(b) the denominator for the Q3 attenuation correction;
(c) the honest restatement of "modest accuracy": relative to the ceiling, not
    relative to 100%.

The 8 repeats of a condition give a natural split-half: 4 + 4 disjoint repeats,
averaged within half, correlated across halves, Spearman-Brown corrected back
up to the full 8-repeat reliability.  Because condition repeats are nested
within subject, all splits are formed **within subject** -- never pooled.

Two flavours of reliability are reported and they answer different questions:

* ``pattern`` reliability -- for each condition, correlate half A's pattern with
  half B's pattern across features (channels x time).  High when a condition's
  spatiotemporal response shape is stable.  This is the quantity that upper
  bounds *pattern-based* readouts (RSA, decoding, retrieval).
* ``feature`` reliability -- for each feature, correlate across conditions.
  High when a channel/timepoint reliably discriminates conditions.  This is the
  quantity that upper bounds *univariate* condition-difference claims.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .retrieval import (
    DEFAULT_GALLERY_SIZES,
    build_pseudo_trials,
    l2_normalize,
    retrieval_metrics,
    to_numpy,
)

__all__ = [
    "spearman_brown",
    "attenuation_correct",
    "fraction_of_ceiling",
    "split_half_reliability",
    "subject_noise_ceiling",
    "condition_noise_ceiling",
    "retrieval_noise_ceiling",
    "noise_ceiling_table",
]

ArrayLike = Any


# --------------------------------------------------------------------------- #
# scalar corrections
# --------------------------------------------------------------------------- #
def spearman_brown(r: ArrayLike, n: float = 2.0) -> np.ndarray | float:
    """Spearman-Brown prophecy: reliability of a test ``n`` times longer.

    ``n=2`` converts a split-half (4-vs-4 repeats) correlation into the
    reliability of the full 8-repeat average.  Negative or degenerate inputs are
    returned as-is after clipping the denominator away from zero, and a NaN is
    produced when the formula is undefined (r = -1/(n-1)).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    denom = 1.0 + (n - 1.0) * r_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (n * r_arr) / denom
    out = np.where(np.abs(denom) < 1e-12, np.nan, out)
    out = np.clip(out, -1.0, 1.0)
    return float(out) if np.ndim(r) == 0 else out


def attenuation_correct(
    r_obs: ArrayLike,
    rel_x: ArrayLike,
    rel_y: ArrayLike = 1.0,
    *,
    clip: bool = True,
) -> Tuple[np.ndarray | float, np.ndarray | float]:
    """Classical disattenuation ``r / sqrt(rel_x * rel_y)``.

    Returns ``(r_corrected, flag)`` where ``flag`` is True wherever the
    correction was unstable (reliability <= 0, or the corrected value exceeded
    1 before clipping).  Unstable corrections must be reported as such, not
    silently clipped, which is why the flag comes back with the value.
    """
    r_arr = np.asarray(r_obs, dtype=np.float64)
    rx = np.asarray(rel_x, dtype=np.float64)
    ry = np.asarray(rel_y, dtype=np.float64)
    denom = np.sqrt(np.clip(rx, 1e-12, None) * np.clip(ry, 1e-12, None))
    bad = (rx <= 0) | (ry <= 0) | ~np.isfinite(denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r_arr / denom
    overshoot = np.abs(out) > 1.0
    flag = bad | overshoot
    if clip:
        out = np.clip(out, -1.0, 1.0)
    out = np.where(bad, np.nan, out)
    if np.ndim(r_obs) == 0 and np.ndim(rel_x) == 0 and np.ndim(rel_y) == 0:
        return float(out), bool(flag)
    return out, flag


def fraction_of_ceiling(
    observed: ArrayLike,
    ceiling: ArrayLike,
    chance: ArrayLike = 0.0,
    *,
    clip: bool = False,
) -> np.ndarray | float:
    """Express a score as a fraction of the attainable range above chance.

        (observed - chance) / (ceiling - chance)

    Works for accuracies (pass the appropriate ``chance``, e.g. 1/18) and for
    correlations (``chance=0``).  Returns NaN when the ceiling is at or below
    chance -- in that case the data cannot support the claim at all and the
    honest report is "ceiling indistinguishable from chance", not a ratio.
    """
    obs = np.asarray(observed, dtype=np.float64)
    ceil = np.asarray(ceiling, dtype=np.float64)
    ch = np.asarray(chance, dtype=np.float64)
    span = ceil - ch
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (obs - ch) / span
    out = np.where(span <= 1e-9, np.nan, out)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return float(out) if np.ndim(observed) == 0 else out


# --------------------------------------------------------------------------- #
# split-half machinery
# --------------------------------------------------------------------------- #
def _flatten_patterns(x: np.ndarray) -> np.ndarray:
    """(n, ...) -> (n, F)."""
    x = np.asarray(x)
    return x.reshape(x.shape[0], -1)


def _corr_rows(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Row-wise correlation between two (n, F) matrices -> (n,)."""
    if method == "spearman":
        a = np.apply_along_axis(stats.rankdata, 1, a)
        b = np.apply_along_axis(stats.rankdata, 1, b)
    elif method != "pearson":
        raise ValueError(f"unknown method {method!r}")
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan)


def split_half_reliability(
    x: ArrayLike,
    item_ids: ArrayLike,
    *,
    subject_ids: Optional[ArrayLike] = None,
    k: int = 4,
    n_splits: int = 100,
    method: str = "pearson",
    seed: int = 0,
    return_per_item: bool = False,
) -> Dict[str, Any]:
    """Split-half reliability of condition patterns from repeated trials.

    Not to be confused with :func:`tactus.common.split_half_reliability`, which
    is the scalar two-vector helper (correlate one pair of half-averages).  This
    function does the resampling: it forms many disjoint 4-vs-4 splits from the
    raw trials and returns both the pattern-wise and feature-wise reliabilities.

    Parameters
    ----------
    x : (n_trials, ...) trial patterns.  Anything past the first axis is
        flattened, so ``(n_trials, 64, 120)`` epochs and ``(n_trials, D)``
        embeddings are both accepted.
    item_ids : (n_trials,) condition id (or video id) per trial.
    subject_ids : (n_trials,) subject id; splits are always formed within
        subject.
    k : repeats per half (4 of the available 8).
    n_splits : independent random 4-vs-4 splits, averaged.

    Returns
    -------
    dict with ``pattern_r``, ``pattern_r_sb``, ``feature_r``, ``feature_r_sb``
    (means over splits), their split-to-split SDs, ``n_items``, ``n_splits``
    and, when ``return_per_item``, the per-item vectors.
    """
    rng = np.random.default_rng(seed)
    xf = _flatten_patterns(to_numpy(x)).astype(np.float64, copy=False)
    items = np.asarray(to_numpy(item_ids)).ravel()
    subs = (
        np.asarray(to_numpy(subject_ids)).ravel()
        if subject_ids is not None
        else np.full(items.shape[0], -1)
    )

    pattern_per_split: List[float] = []
    feature_per_split: List[float] = []
    per_item_acc: Dict[Any, List[float]] = {}
    n_items_used = 0

    for _ in range(n_splits):
        a, b, split_items, _ = build_pseudo_trials(
            xf, items, k=k, subject_ids=subs, rng=rng,
            renormalize=False, disjoint_halves=True,
        )
        if a.shape[0] < 3:
            continue
        n_items_used = max(n_items_used, int(np.unique(split_items).size))

        r_items = _corr_rows(a, b, method=method)
        pattern_per_split.append(float(np.nanmean(r_items)))
        if return_per_item:
            for it, r in zip(split_items, r_items):
                per_item_acc.setdefault(it, []).append(float(r))

        # feature-wise: correlate across items, per feature
        r_feats = _corr_rows(a.T, b.T, method=method)
        feature_per_split.append(float(np.nanmean(r_feats)))

    if not pattern_per_split:
        warnings.warn(
            "split_half_reliability: no (subject, item) group had >= 2k trials; "
            "returning NaNs",
            RuntimeWarning,
        )
        return {
            "pattern_r": np.nan, "pattern_r_sb": np.nan, "pattern_r_sd": np.nan,
            "feature_r": np.nan, "feature_r_sb": np.nan, "feature_r_sd": np.nan,
            "n_items": 0, "n_splits": 0, "k": k, "method": method,
        }

    pat = float(np.nanmean(pattern_per_split))
    fea = float(np.nanmean(feature_per_split))
    out: Dict[str, Any] = {
        "pattern_r": pat,
        "pattern_r_sb": float(spearman_brown(pat, 2.0)),
        "pattern_r_sd": float(np.nanstd(pattern_per_split, ddof=1))
        if len(pattern_per_split) > 1 else np.nan,
        "feature_r": fea,
        "feature_r_sb": float(spearman_brown(fea, 2.0)),
        "feature_r_sd": float(np.nanstd(feature_per_split, ddof=1))
        if len(feature_per_split) > 1 else np.nan,
        "n_items": n_items_used,
        "n_splits": len(pattern_per_split),
        "k": k,
        "method": method,
    }
    if return_per_item:
        out["per_item_r"] = {
            it: float(np.nanmean(v)) for it, v in per_item_acc.items()
        }
        out["per_item_r_sb"] = {
            it: float(spearman_brown(np.nanmean(v), 2.0))
            for it, v in per_item_acc.items()
        }
    return out


def subject_noise_ceiling(
    x: ArrayLike,
    item_ids: ArrayLike,
    subject_ids: ArrayLike,
    *,
    k: int = 4,
    n_splits: int = 50,
    method: str = "pearson",
    seed: int = 0,
) -> pd.DataFrame:
    """Per-subject split-half reliability (one row per subject).

    Columns: ``subject_id, pattern_r, pattern_r_sb, feature_r, feature_r_sb,
    n_items, n_trials``.  ``pattern_r_sb`` is the number to use as the Q3
    covariate "data quality" and as the retrieval-ceiling proxy when the
    empirical retrieval ceiling is too expensive to compute.
    """
    xf = _flatten_patterns(to_numpy(x))
    items = np.asarray(to_numpy(item_ids)).ravel()
    subs = np.asarray(to_numpy(subject_ids)).ravel()
    rows: List[Dict[str, Any]] = []
    for i_sub, s in enumerate(np.unique(subs)):
        sel = np.flatnonzero(subs == s)
        # seed from the enumeration index, not hash(str(s)): Python string
        # hashing is salted per process, which would make runs irreproducible
        res = split_half_reliability(
            xf[sel], items[sel], subject_ids=subs[sel], k=k,
            n_splits=n_splits, method=method, seed=seed + 1000 * i_sub,
        )
        rows.append(
            {
                "subject_id": s,
                "pattern_r": res["pattern_r"],
                "pattern_r_sb": res["pattern_r_sb"],
                "feature_r": res["feature_r"],
                "feature_r_sb": res["feature_r_sb"],
                "n_items": res["n_items"],
                "n_trials": int(sel.size),
            }
        )
    return pd.DataFrame(rows)


def condition_noise_ceiling(
    x: ArrayLike,
    item_ids: ArrayLike,
    subject_ids: ArrayLike,
    *,
    k: int = 4,
    n_splits: int = 50,
    method: str = "pearson",
    seed: int = 0,
) -> pd.DataFrame:
    """Per-condition reliability, averaged over subjects (one row per item).

    Identifies stimuli that simply do not drive a reliable response; those
    conditions cap the video-level analyses and should be reported alongside
    any per-video effect, otherwise a null for a given video is uninterpretable.
    """
    xf = _flatten_patterns(to_numpy(x))
    items = np.asarray(to_numpy(item_ids)).ravel()
    subs = np.asarray(to_numpy(subject_ids)).ravel()

    acc: Dict[Any, List[float]] = {}
    for i_sub, s in enumerate(np.unique(subs)):
        sel = np.flatnonzero(subs == s)
        res = split_half_reliability(
            xf[sel], items[sel], subject_ids=subs[sel], k=k, n_splits=n_splits,
            method=method, seed=seed + 1000 * i_sub,
            return_per_item=True,
        )
        for it, r in res.get("per_item_r", {}).items():
            acc.setdefault(it, []).append(r)

    rows = []
    for it, vals in acc.items():
        m = float(np.nanmean(vals))
        rows.append(
            {
                "item_id": it,
                "pattern_r": m,
                "pattern_r_sb": float(spearman_brown(m, 2.0)),
                "n_subjects": int(len(vals)),
                "sd_across_subjects": float(np.nanstd(vals, ddof=1))
                if len(vals) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the ceiling that is actually in retrieval units
# --------------------------------------------------------------------------- #
def retrieval_noise_ceiling(
    z: ArrayLike,
    item_ids: ArrayLike,
    subject_ids: ArrayLike,
    *,
    k: int = 4,
    k_gallery: Optional[int] = None,
    n_resamples: int = 50,
    gallery_sizes: Sequence[int] = DEFAULT_GALLERY_SIZES,
    groups: Optional[ArrayLike] = None,
    item_group_ids: Optional[ArrayLike] = None,
    seed: int = 0,
    per_subject: bool = True,
    n_gallery_subjects: Optional[int] = None,
    n_gallery_draws: int = 20,
) -> pd.DataFrame:
    """EEG->EEG split-half retrieval: a ceiling *in the endpoint's own units*.

    Query = pseudo-trial from ``k`` repeats, gallery = pseudo-trial from
    ``k_gallery`` disjoint repeats (default ``k``), one gallery row per item.

    **What this number is and is not.**  Both sides carry k-repeat noise, so it
    is a *reliability-matched* ceiling, not a hard upper bound:

    * it **underestimates** the ceiling for retrieval against the frozen video
      embedding, whose gallery is noiseless -- a model can legitimately exceed
      this number, and observing that is informative rather than paradoxical;
    * it **overestimates** what a model generalising across subjects can reach,
      because query and gallery come from the same participant;
    * it **underestimates** what a model averaging more than ``k`` repeats
      could reach.

    Set ``k=1, k_gallery=7`` for the single-trial query against the cleanest
    gallery the 8 repeats can build; that variant is the closest available
    approximation to a true single-trial ceiling, and the gap between it and the
    ``k=4, k_gallery=4`` variant is itself the reliability report.

    Parameters
    ----------
    item_group_ids : optional (n_unique_items,) group label per *item* aligned
        to ``np.unique(item_ids)``; enables the within-material ceiling.
    """
    rng = np.random.default_rng(seed)
    k_gal = int(k_gallery if k_gallery is not None else k)
    z_arr = _flatten_patterns(to_numpy(z)).astype(np.float64, copy=False)
    items = np.asarray(to_numpy(item_ids)).ravel()
    subs = np.asarray(to_numpy(subject_ids)).ravel()

    # The "pooled" gallery averages each item over EVERY subject in the input, so
    # its cleanliness -- and therefore the ceiling -- scales with how many
    # subjects the caller happened to pass.  Measured on one within_subject fold:
    # 10/20/40/80 subjects give 0.1133/0.1317/0.1497/0.1633.  A within_subject
    # fold carries 80 test subjects and a double_disjoint fold carries 10, so
    # comparing "fraction of ceiling" across those two regimes compares two
    # different denominators and is meaningless (DECISIONS D15).
    #
    # ``n_gallery_subjects`` pins the count so one common denominator can be used
    # everywhere.  The per-subject rows are unaffected: they never pool.
    #
    # Pinning the count is necessary but not sufficient, and this cost a wrong
    # comparison before it was measured: *which* 10 subjects are drawn moves the
    # pooled ceiling from 0.1122 to 0.1539 across eight seeds on one fold
    # (sd 0.0143 on a mean of 0.1272).  Two arms drawing different subsets
    # therefore get denominators differing by more than the accuracy gap being
    # compared.  So the pooled ceiling is averaged over ``n_gallery_draws``
    # independent subsets and its spread is carried alongside, and any
    # fraction-of-ceiling comparison narrower than that spread is unresolvable.
    pool_sets: List[np.ndarray] = [np.arange(items.shape[0])]
    if n_gallery_subjects is not None:
        uniq_subs = np.unique(subs)
        if uniq_subs.size > int(n_gallery_subjects):
            draw_rng = np.random.default_rng(seed)
            pool_sets = []
            for _ in range(max(1, int(n_gallery_draws))):
                keep = draw_rng.choice(uniq_subs, size=int(n_gallery_subjects),
                                       replace=False)
                pool_sets.append(np.flatnonzero(np.isin(subs, keep)))
    subject_sets: List[Tuple[Any, np.ndarray]] = [
        (("pooled" if len(pool_sets) == 1 else f"pooled_draw{i:02d}"), idx)
        for i, idx in enumerate(pool_sets)
    ]
    if per_subject:
        subject_sets += [(s, np.flatnonzero(subs == s)) for s in np.unique(subs)]

    frames: List[pd.DataFrame] = []
    for subject_key, sel in subject_sets:
        if sel.size < k + k_gal:
            continue
        per_resample: List[Dict[str, float]] = []
        for _ in range(n_resamples):
            a, b, split_items, _ = build_pseudo_trials(
                z_arr[sel], items[sel], k=k, k_second=k_gal,
                subject_ids=subs[sel], rng=rng,
                renormalize=True, disjoint_halves=True,
            )
            if a.shape[0] < 2:
                continue
            # one gallery row per unique item (pooled mode may repeat items
            # across subjects; average them so the gallery stays item-indexed)
            uniq, inv = np.unique(split_items, return_inverse=True)
            gal = np.zeros((uniq.size, b.shape[1]), dtype=np.float64)
            np.add.at(gal, inv, b)
            counts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
            gal = l2_normalize(gal / counts[:, None])
            grp_arr = None
            if item_group_ids is not None:
                gmap = dict(zip(np.unique(items), np.asarray(to_numpy(item_group_ids)).ravel()))
                grp_arr = np.array([gmap.get(u, -1) for u in uniq])
            elif groups is not None:
                g_all = np.asarray(to_numpy(groups)).ravel()[sel]
                grp_arr = np.array(
                    [g_all[np.flatnonzero(split_items == u)[0]] for u in uniq]
                )
            per_resample.append(
                retrieval_metrics(
                    l2_normalize(a), gal, inv, grp_arr, gallery_sizes=gallery_sizes
                )
            )
        if not per_resample:
            continue
        keys = sorted(set().union(*[set(d) for d in per_resample]))
        mean_metrics = {
            key: float(np.nanmean([d[key] for d in per_resample if key in d]))
            for key in keys
        }
        rows = []
        for key, val in mean_metrics.items():
            if key in ("n_queries", "n_items") or "chance" in key:
                continue
            rows.append(
                {
                    "subject_id": subject_key,
                    "endpoint": key,
                    "ceiling": val,
                    "n_resamples": len(per_resample),
                    "k_query": k,
                    "k_gallery": k_gal,
                    "n_items": mean_metrics.get("n_items", np.nan),
                }
            )
        frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["subject_id", "endpoint", "ceiling",
                                     "n_resamples", "k_query", "k_gallery",
                                     "n_items"])
    out = pd.concat(frames, ignore_index=True)

    # Collapse the per-draw pooled rows into one, carrying the spread.  Callers
    # read the row labelled "pooled"; without this they would silently read
    # draw 00 and inherit its sampling noise as if it were the ceiling.
    draws = out[out["subject_id"].astype(str).str.startswith("pooled_draw")]
    if len(draws):
        agg = (draws.groupby("endpoint", as_index=False)
                    .agg(ceiling=("ceiling", "mean"),
                         ceiling_sd=("ceiling", "std"),
                         ceiling_lo=("ceiling", "min"),
                         ceiling_hi=("ceiling", "max"),
                         n_resamples=("n_resamples", "sum"),
                         k_query=("k_query", "first"),
                         k_gallery=("k_gallery", "first"),
                         n_items=("n_items", "first")))
        agg.insert(0, "subject_id", "pooled")
        agg["n_gallery_draws"] = int(draws["subject_id"].nunique())
        out = pd.concat([agg, out[~out.index.isin(draws.index)]], ignore_index=True)
    return out


def noise_ceiling_table(
    observed_long: pd.DataFrame,
    ceiling_long: pd.DataFrame,
    *,
    endpoint_col: str = "endpoint",
    value_col: str = "value",
    chance_col: str = "chance",
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Join observed scores to ceilings and add the fraction-of-ceiling column.

    ``observed_long`` is expected to carry one row per (subject, endpoint) with
    the observed value and its chance level -- e.g. the output of
    :func:`tactus.eval.retrieval.evaluate_retrieval` after adding an
    ``endpoint`` column built from ``gallery`` + ``metric``.
    """
    obs = observed_long.copy()
    if endpoint_col not in obs.columns and {"gallery", "metric"} <= set(obs.columns):
        obs[endpoint_col] = obs["gallery"].astype(str) + "_" + obs["metric"].astype(str)
    merged = obs.merge(
        ceiling_long.rename(columns={"ceiling": "_ceiling"}),
        on=[subject_col, endpoint_col],
        how="left",
        suffixes=("", "_ceil"),
    )
    merged["fraction_of_ceiling"] = fraction_of_ceiling(
        merged[value_col].to_numpy(dtype=float),
        merged["_ceiling"].to_numpy(dtype=float),
        merged[chance_col].to_numpy(dtype=float)
        if chance_col in merged.columns else 0.0,
    )
    merged["ceiling"] = merged.pop("_ceiling")
    chance_vals = (
        merged[chance_col].to_numpy(dtype=float)
        if chance_col in merged.columns else np.zeros(len(merged))
    )
    merged["ceiling_above_chance"] = merged["ceiling"].to_numpy(dtype=float) > chance_vals
    # A model retrieving against a NOISELESS video gallery can legitimately beat
    # the reliability-matched EEG->EEG ceiling; flag it rather than clipping, so
    # the report says "exceeds the reliability-matched ceiling" instead of
    # printing a fraction above 1 with no explanation.
    merged["exceeds_ceiling"] = (
        merged[value_col].to_numpy(dtype=float) > merged["ceiling"].to_numpy(dtype=float)
    )
    return merged
