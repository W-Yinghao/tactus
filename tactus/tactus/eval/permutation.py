"""Permutation inference with the correct exchangeable unit: the BASE VIDEO.

Why this module exists
----------------------
In ds005662 each base video contributes 4 orientations x 8 repeats = 32 trials
*per subject*.  Those 32 trials are a cluster: they share the stimulus, and
therefore share whatever stimulus-driven signal the model is being tested for.
Permuting *trial* labels destroys the cluster structure and yields a null
distribution that is 4-9x too narrow -- an institutionalised false positive
machine.  The exchangeable unit is the base video (90 units), and the thing we
shuffle is the **video -> embedding assignment**, never the trial labels.

Everything here therefore takes a permutation of the 90 base videos and maps it
onto whatever indexing the statistic needs (conditions, trials, RDM rows).

The (wrong) trial-level null is implemented too -- deliberately, and named
``trial_level_null_diagnostic`` -- so the paper can *show* the narrowing factor
instead of asserting it.  It must never be used for a reported p-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .retrieval import l2_normalize, retrieval_metrics, to_numpy

__all__ = [
    "PermutationResult",
    "permutation_pvalue",
    "sample_base_permutation",
    "condition_permutation_from_base",
    "trial_permutation_from_base",
    "video_level_permutation_test",
    "trial_level_null_diagnostic",
    "null_narrowing_report",
    "retrieval_permutation_test",
    "maxstat_correction",
]

ArrayLike = Any
N_BASE_VIDEOS = 90
N_ORIENTATIONS = 4
N_CONDITIONS = N_BASE_VIDEOS * N_ORIENTATIONS


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class PermutationResult:
    """Outcome of a permutation test, with everything the report needs."""

    observed: float
    null: np.ndarray
    p_value: float
    null_mean: float
    null_sd: float
    null_q: Dict[str, float]
    z_score: float
    n_perm: int
    unit: str
    n_units: int
    tail: str = "greater"
    statistic_name: str = "statistic"
    strata: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """Flat, report-ready dict (the null array itself is not included)."""
        return {
            "statistic": self.statistic_name,
            "observed": self.observed,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "null_p95": self.null_q.get("q95", float("nan")),
            "null_p99": self.null_q.get("q99", float("nan")),
            "z_score": self.z_score,
            "n_perm": self.n_perm,
            "exchangeable_unit": self.unit,
            "n_units": self.n_units,
            "tail": self.tail,
            "strata": self.strata,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary()])


def permutation_pvalue(
    observed: float, null: np.ndarray, tail: str = "greater"
) -> float:
    """(1 + #{null at least as extreme}) / (1 + n_perm) -- never returns 0."""
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    n = null.size
    if n == 0:
        return float("nan")
    if tail == "greater":
        count = int(np.sum(null >= observed))
    elif tail == "less":
        count = int(np.sum(null <= observed))
    elif tail == "two-sided":
        centre = float(np.median(null))
        count = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    else:
        raise ValueError(f"unknown tail {tail!r}")
    return float((1 + count) / (1 + n))


def _finalise(
    observed: float,
    null: np.ndarray,
    *,
    tail: str,
    unit: str,
    n_units: int,
    statistic_name: str,
    strata: Optional[str] = None,
) -> PermutationResult:
    null = np.asarray(null, dtype=np.float64)
    finite = null[np.isfinite(null)]
    sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
    mean = float(np.mean(finite)) if finite.size else float("nan")
    z = float((observed - mean) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    q = {
        "q05": float(np.quantile(finite, 0.05)) if finite.size else float("nan"),
        "q50": float(np.quantile(finite, 0.50)) if finite.size else float("nan"),
        "q95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "q99": float(np.quantile(finite, 0.99)) if finite.size else float("nan"),
    }
    return PermutationResult(
        observed=float(observed),
        null=null,
        p_value=permutation_pvalue(observed, null, tail),
        null_mean=mean,
        null_sd=sd,
        null_q=q,
        z_score=z,
        n_perm=int(null.size),
        unit=unit,
        n_units=int(n_units),
        tail=tail,
        statistic_name=statistic_name,
        strata=strata,
    )


# --------------------------------------------------------------------------- #
# building permutations
# --------------------------------------------------------------------------- #
def sample_base_permutation(
    n_base: int = N_BASE_VIDEOS,
    *,
    rng: Optional[np.random.Generator] = None,
    strata: Optional[ArrayLike] = None,
) -> np.ndarray:
    """A permutation of base-video indices ``0..n_base-1``.

    ``strata`` (e.g. the 8-way material label per base video) restricts the
    shuffle to *within* stratum.  A stratified null answers the sharper
    question "is there video-specific information beyond the attribute?", which
    is exactly the attribute-shortcut concern in BLUEPRINT v2 §5.2.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    perm = np.arange(n_base)
    if strata is None:
        rng.shuffle(perm)
        return perm
    s = np.asarray(to_numpy(strata)).ravel()
    if s.shape[0] != n_base:
        raise ValueError("strata must have one label per base video")
    for level in np.unique(s):
        idx = np.flatnonzero(s == level)
        perm[idx] = rng.permutation(idx)
    return perm


def condition_permutation_from_base(
    perm_base: np.ndarray,
    *,
    video_of_condition: Optional[np.ndarray] = None,
    orientation_of_condition: Optional[np.ndarray] = None,
    n_conditions: int = N_CONDITIONS,
) -> np.ndarray:
    """Lift a base-video permutation to a permutation of condition ids.

    Default indexing follows the shared contract:
    ``condition_id = (video_id - 1) * 4 + orientation``, so
    ``video_of_condition = condition_id // 4`` and
    ``orientation_of_condition = condition_id % 4``.

    Returns ``src`` such that ``permuted[c] = original[src[c]]``: the four
    orientations of a base video move together, preserving the orientation
    structure while destroying the video identity mapping.
    """
    perm_base = np.asarray(perm_base, dtype=np.int64)
    if video_of_condition is None:
        video_of_condition = np.arange(n_conditions) // N_ORIENTATIONS
    if orientation_of_condition is None:
        orientation_of_condition = np.arange(n_conditions) % N_ORIENTATIONS
    v = np.asarray(video_of_condition, dtype=np.int64)
    o = np.asarray(orientation_of_condition, dtype=np.int64)
    if v.shape != o.shape:
        raise ValueError("video_of_condition and orientation_of_condition must align")

    # map (video, orientation) -> condition index, then reindex by perm_base
    n_or = int(o.max()) + 1
    lookup = np.full((int(v.max()) + 1, n_or), -1, dtype=np.int64)
    lookup[v, o] = np.arange(v.shape[0])
    src = lookup[perm_base[v], o]
    if np.any(src < 0):
        raise ValueError(
            "condition grid is ragged: some (video, orientation) pairs are missing; "
            "pass explicit video_of_condition / orientation_of_condition arrays"
        )
    return src


def trial_permutation_from_base(
    perm_base: np.ndarray,
    trial_video_id: ArrayLike,
    *,
    video_id_base: int = 1,
) -> np.ndarray:
    """Relabel each trial's base video under ``perm_base`` (0-based internally).

    Returns the *surrogate* base-video id per trial.  Trials keep their EEG,
    their trial index and their subject; only the stimulus identity attached to
    them is shuffled, and it is shuffled coherently for all 32 trials of a video.
    """
    perm_base = np.asarray(perm_base, dtype=np.int64)
    v = np.asarray(to_numpy(trial_video_id), dtype=np.int64).ravel() - video_id_base
    if v.min() < 0 or v.max() >= perm_base.shape[0]:
        raise ValueError("trial_video_id out of range for perm_base")
    return perm_base[v] + video_id_base


# --------------------------------------------------------------------------- #
# the tests
# --------------------------------------------------------------------------- #
def video_level_permutation_test(
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_base: int = N_BASE_VIDEOS,
    n_perm: int = 1000,
    seed: int = 0,
    strata: Optional[ArrayLike] = None,
    observed: Optional[float] = None,
    tail: str = "greater",
    statistic_name: str = "statistic",
    progress: Optional[Callable[[int, int], None]] = None,
) -> PermutationResult:
    """Permutation test whose exchangeable unit is the base video.

    Parameters
    ----------
    stat_fn : callable taking a base-video permutation (``(n_base,)`` int array,
        0-based) and returning a scalar statistic.  Called once with the
        identity permutation to get the observed value unless ``observed`` is
        supplied.
    strata : optional per-base-video labels restricting the shuffle to within
        stratum (material-matched null).
    """
    rng = np.random.default_rng(seed)
    identity = np.arange(n_base)
    obs = float(stat_fn(identity)) if observed is None else float(observed)

    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = sample_base_permutation(n_base, rng=rng, strata=strata)
        null[i] = float(stat_fn(perm))
        if progress is not None and (i % 50 == 0):
            progress(i, n_perm)

    strata_desc = None
    if strata is not None:
        s = np.asarray(to_numpy(strata)).ravel()
        strata_desc = f"within-stratum ({len(np.unique(s))} levels)"
    return _finalise(
        obs, null, tail=tail, unit="base_video", n_units=n_base,
        statistic_name=statistic_name, strata=strata_desc,
    )


def trial_level_null_diagnostic(
    stat_fn: Callable[[np.ndarray], float],
    n_trials: int,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    observed: Optional[float] = None,
    tail: str = "greater",
    statistic_name: str = "statistic",
) -> PermutationResult:
    """The WRONG null: shuffle trial labels, ignoring the base-video clusters.

    **Never report a p-value from this function.**  It exists so the paper can
    quantify how much narrower the naive null would have been (see
    :func:`null_narrowing_report`).  ``stat_fn`` receives a permutation of trial
    indices.
    """
    rng = np.random.default_rng(seed)
    obs = float(stat_fn(np.arange(n_trials))) if observed is None else float(observed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = float(stat_fn(rng.permutation(n_trials)))
    res = _finalise(
        obs, null, tail=tail, unit="trial(INVALID)", n_units=n_trials,
        statistic_name=statistic_name,
    )
    return res


def null_narrowing_report(
    video_result: PermutationResult,
    trial_result: PermutationResult,
) -> Dict[str, float]:
    """How much narrower is the (invalid) trial-level null?

    Returns the SD ratio, the two 95th percentiles, the p-value the trial-level
    null would have produced for the *same* observed statistic, and the
    resulting "significance inflation" summary line used in REPORT.md.
    """
    sd_video = video_result.null_sd
    sd_trial = trial_result.null_sd
    ratio = float(sd_video / sd_trial) if sd_trial and sd_trial > 0 else float("nan")
    p_wrong = permutation_pvalue(
        video_result.observed, trial_result.null, tail=video_result.tail
    )
    return {
        "null_sd_video_level": sd_video,
        "null_sd_trial_level": sd_trial,
        "narrowing_factor": ratio,
        "null_p95_video_level": video_result.null_q.get("q95", float("nan")),
        "null_p95_trial_level": trial_result.null_q.get("q95", float("nan")),
        "p_value_video_level": video_result.p_value,
        "p_value_if_trial_level_null": p_wrong,
        "observed": video_result.observed,
    }


# --------------------------------------------------------------------------- #
# ready-made statistic: retrieval accuracy under a shuffled video->embedding map
# --------------------------------------------------------------------------- #
def retrieval_permutation_test(
    z_eeg: ArrayLike,
    trial_item_id: ArrayLike,
    gallery_emb: ArrayLike,
    gallery_item_ids: ArrayLike,
    *,
    item_kind: str = "video",
    metric: str = "nway18_top1",
    groups: Optional[ArrayLike] = None,
    subject_ids: Optional[ArrayLike] = None,
    pseudo_k: Optional[int] = 4,
    n_perm: int = 1000,
    seed: int = 0,
    strata: Optional[ArrayLike] = None,
    also_trial_level: bool = True,
    n_perm_trial_level: Optional[int] = None,
) -> Dict[str, Any]:
    """Permutation test for a retrieval endpoint, shuffling video -> embedding.

    The gallery rows are permuted at the base-video level (all four orientations
    of a video move together when ``item_kind == "condition"``).  The EEG side,
    the trial ordering and the subject structure are untouched, so anything the
    model learned from trial position or subject identity survives the shuffle
    -- which is the point: the null asks specifically whether *stimulus
    identity* is being recovered.

    Parameters
    ----------
    item_kind : ``"video"`` (gallery has 90 rows, ids 1..90) or ``"condition"``
        (gallery has 360 rows, ids 0..359 with ``(video-1)*4 + orientation``).
    metric : key of :func:`~tactus.eval.retrieval.retrieval_metrics` to test.
    pseudo_k : if not None, statistics are computed on k-averaged pseudo-trials
        (a single fixed draw, reused across permutations so that the only thing
        varying is the shuffle).
    strata : per-base-video stratification labels (e.g. material), giving the
        material-matched null.

    Returns
    -------
    dict with ``"video_level"`` (:class:`PermutationResult`),
    ``"trial_level"`` (:class:`PermutationResult` or None) and ``"narrowing"``.
    """
    from .retrieval import build_pseudo_trials  # local import: avoid cycles at import time

    rng = np.random.default_rng(seed)
    z = l2_normalize(to_numpy(z_eeg))
    g_emb = l2_normalize(to_numpy(gallery_emb))
    g_ids = np.asarray(to_numpy(gallery_item_ids)).ravel()
    t_ids = np.asarray(to_numpy(trial_item_id)).ravel()
    subs = (
        np.asarray(to_numpy(subject_ids)).ravel()
        if subject_ids is not None
        else np.full(t_ids.shape[0], -1)
    )
    grp = np.asarray(to_numpy(groups)).ravel() if groups is not None else None

    if pseudo_k:
        z, t_ids, subs = build_pseudo_trials(
            z, t_ids, k=pseudo_k, subject_ids=subs, rng=rng
        )

    order = np.argsort(g_ids)
    g_ids_sorted = g_ids[order]
    true_idx = order[np.searchsorted(g_ids_sorted, t_ids)]

    # base-video index of each gallery row
    if item_kind == "video":
        base_of_row = g_ids.astype(np.int64) - 1
        orient_of_row = np.zeros_like(base_of_row)
    elif item_kind == "condition":
        base_of_row = g_ids.astype(np.int64) // 4
        orient_of_row = g_ids.astype(np.int64) % 4
    else:
        raise ValueError("item_kind must be 'video' or 'condition'")
    n_base = int(base_of_row.max()) + 1

    def _stat_from_gallery(gal: np.ndarray) -> float:
        m = retrieval_metrics(z, gal, true_idx, grp)
        return float(m.get(metric, np.nan))

    def stat_fn(perm_base: np.ndarray) -> float:
        src = condition_permutation_from_base(
            perm_base,
            video_of_condition=base_of_row,
            orientation_of_condition=orient_of_row,
            n_conditions=g_emb.shape[0],
        )
        return _stat_from_gallery(g_emb[src])

    video_res = video_level_permutation_test(
        stat_fn, n_base=n_base, n_perm=n_perm, seed=seed, strata=strata,
        statistic_name=metric,
    )

    trial_res: Optional[PermutationResult] = None
    narrowing: Optional[Dict[str, float]] = None
    if also_trial_level:
        n_tl = n_perm_trial_level or n_perm

        def stat_fn_trial(perm_trials: np.ndarray) -> float:
            m = retrieval_metrics(z, g_emb, true_idx[perm_trials], grp)
            return float(m.get(metric, np.nan))

        trial_res = trial_level_null_diagnostic(
            stat_fn_trial, z.shape[0], n_perm=n_tl, seed=seed + 1,
            observed=video_res.observed, statistic_name=metric,
        )
        narrowing = null_narrowing_report(video_res, trial_res)

    return {"video_level": video_res, "trial_level": trial_res, "narrowing": narrowing}


# --------------------------------------------------------------------------- #
# family-wise correction from a shared permutation family
# --------------------------------------------------------------------------- #
def maxstat_correction(
    observed: Dict[str, float],
    nulls: Dict[str, np.ndarray],
    *,
    tail: str = "greater",
) -> pd.DataFrame:
    """Westfall-Young max-statistic correction across an endpoint family.

    All entries of ``nulls`` must come from the *same* permutations in the same
    order (i.e. the same ``seed`` and the same number of permutations), so that
    the row-wise maximum preserves the dependence between endpoints.  Used for
    the Q2 time-course feature-layer max-stat in BLUEPRINT v2 §5.2.
    """
    keys = list(observed)
    missing = [k for k in keys if k not in nulls]
    if missing:
        raise ValueError(f"no null distribution for endpoints: {missing}")
    lengths = {len(np.asarray(nulls[k]).ravel()) for k in keys}
    if len(lengths) != 1:
        raise ValueError("all nulls must share the same number of permutations")

    stack = np.vstack([np.asarray(nulls[k], dtype=np.float64).ravel() for k in keys])
    if tail == "greater":
        max_null = stack.max(axis=0)
    elif tail == "less":
        max_null = stack.min(axis=0)
    else:
        centre = np.median(stack, axis=1, keepdims=True)
        max_null = np.abs(stack - centre).max(axis=0)

    rows = []
    for k in keys:
        obs = float(observed[k])
        if tail == "two-sided":
            centre_k = float(np.median(nulls[k]))
            stat = abs(obs - centre_k)
            p_corr = float((1 + np.sum(max_null >= stat)) / (1 + max_null.size))
        else:
            p_corr = permutation_pvalue(obs, max_null, tail=tail)
        rows.append(
            {
                "endpoint": k,
                "observed": obs,
                "p_uncorrected": permutation_pvalue(obs, nulls[k], tail=tail),
                "p_maxstat": p_corr,
                "n_perm": int(max_null.size),
            }
        )
    return pd.DataFrame(rows).sort_values("p_maxstat").reset_index(drop=True)
