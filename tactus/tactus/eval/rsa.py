"""Time-resolved RSA over the 360 conditions, with partial correlation controls.

What this module computes
-------------------------
1.  **EEG RDMs**, one per timepoint, over the 360 conditions (90 base videos x 4
    orientations).  Two distance estimators:

    * ``crossnobis`` -- cross-validated Mahalanobis (whitened, cross-validated
      across pseudo-trial folds).  Unbiased: its expectation is zero when two
      conditions have identical true patterns, so a flat RDM really means "no
      difference" rather than "noise floor".  Preferred whenever a noise
      covariance can be estimated.
    * ``correlation`` -- 1 - Pearson r across channels.  Fallback; biased upward
      by noise, so its absolute level is uninterpretable (only its *structure*
      is), and it is used for sanity checks and for very short windows.

2.  **Model RDMs**: video-embedding (cosine), attribute (categorical mismatch or
    absolute difference), and low-level motion-energy / luminance / contrast.

3.  **Spearman and partial-Spearman** between EEG and model RDMs, partialling
    out the low-level RDM, with **video-level permutation** inference and a
    cluster-based correction over time.

Permutation scheme
------------------
Only the *model* RDM is permuted, and it is permuted by shuffling the 90 base
videos (all four orientations of a video move together).  The EEG RDM and the
low-level control RDM keep their alignment, so the test asks the incremental
question "does video-embedding structure explain EEG structure beyond low-level
structure?".  Ranks are computed once and gathered under the permutation, which
makes 1000 permutations x 120 timepoints a matter of seconds rather than hours.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .permutation import (
    N_BASE_VIDEOS,
    N_CONDITIONS,
    N_ORIENTATIONS,
    condition_permutation_from_base,
    sample_base_permutation,
)
from .retrieval import l2_normalize, to_numpy

__all__ = [
    "upper_tri_indices",
    "vectorize_rdm",
    "rank_matrix",
    "fold_averaged_patterns",
    "estimate_whitener",
    "crossnobis_rdm",
    "correlation_rdm",
    "time_resolved_rdms",
    "rdm_from_embeddings",
    "rdm_from_categorical",
    "rdm_from_continuous",
    "rdm_from_features",
    "build_model_rdms",
    "spearman_rdm_corr",
    "partial_spearman",
    "rsa_time_course",
    "rsa_noise_ceiling",
]

ArrayLike = Any


# --------------------------------------------------------------------------- #
# RDM plumbing
# --------------------------------------------------------------------------- #
def upper_tri_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached-friendly wrapper around ``np.triu_indices(n, k=1)``."""
    return np.triu_indices(n, k=1)


def vectorize_rdm(rdm: np.ndarray) -> np.ndarray:
    """Upper triangle (excluding diagonal) of a square RDM as a 1-D vector."""
    rdm = np.asarray(rdm)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError("rdm must be square")
    iu = upper_tri_indices(rdm.shape[0])
    return rdm[iu]


def rank_matrix(rdm: np.ndarray) -> np.ndarray:
    """Symmetric matrix whose entries are the ranks of the RDM's pair values.

    Permuting conditions permutes pairs bijectively, so the ranks of a permuted
    RDM equal the permuted ranks of the original.  Ranking once and gathering
    under each permutation is therefore exact, and removes the per-permutation
    sort that would otherwise dominate the runtime.
    """
    rdm = np.asarray(rdm, dtype=np.float64)
    n = rdm.shape[0]
    iu = upper_tri_indices(n)
    ranks = stats.rankdata(rdm[iu])
    out = np.zeros((n, n), dtype=np.float64)
    out[iu] = ranks
    out += out.T
    return out


def _zscore_rows(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-12)


def _residualize_rows(z: np.ndarray, controls: Optional[np.ndarray]) -> np.ndarray:
    """Regress out ``controls`` (P, m) from each row of ``z`` (n, P)."""
    if controls is None or controls.size == 0:
        return z
    c = np.asarray(controls, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    c = np.column_stack([np.ones(c.shape[0]), c])
    gram = c.T @ c
    coef = np.linalg.solve(gram, c.T @ z.T)  # (m+1, n)
    return z - (c @ coef).T


# --------------------------------------------------------------------------- #
# EEG RDMs
# --------------------------------------------------------------------------- #
def fold_averaged_patterns(
    x: ArrayLike,
    cond_ids: ArrayLike,
    *,
    n_folds: int = 4,
    n_conditions: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    min_trials: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split each condition's trials into ``n_folds`` pseudo-trials.

    Parameters
    ----------
    x : (n_trials, C, T) epochs of ONE subject.
    cond_ids : (n_trials,) condition index in ``0..n_conditions-1``.

    Returns
    -------
    patterns : (n_folds, n_conditions, C, T) float32, NaN for conditions with
        too few trials.
    valid : (n_conditions,) boolean mask of usable conditions.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x_arr = np.asarray(to_numpy(x), dtype=np.float32)
    if x_arr.ndim == 2:  # (n_trials, F) -> treat as single timepoint
        x_arr = x_arr[:, :, None]
    c = np.asarray(to_numpy(cond_ids), dtype=np.int64).ravel()
    n_cond = int(n_conditions if n_conditions is not None else c.max() + 1)
    min_trials = min_trials if min_trials is not None else n_folds

    n_ch, n_t = x_arr.shape[1], x_arr.shape[2]
    out = np.full((n_folds, n_cond, n_ch, n_t), np.nan, dtype=np.float32)
    valid = np.zeros(n_cond, dtype=bool)

    order = np.argsort(c, kind="stable")
    c_sorted = c[order]
    bounds = np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1]])
    starts, ends = bounds, np.r_[bounds[1:], c_sorted.size]
    for s, e in zip(starts, ends):
        cid = int(c_sorted[s])
        members = rng.permutation(order[s:e])
        if members.size < min_trials:
            continue
        chunks = np.array_split(members, n_folds)
        if any(ch.size == 0 for ch in chunks):
            continue
        for f, ch in enumerate(chunks):
            out[f, cid] = x_arr[ch].mean(axis=0)
        valid[cid] = True
    return out, valid


def estimate_whitener(
    x: ArrayLike,
    cond_ids: ArrayLike,
    *,
    method: str = "ledoit-wolf",
    pool_time: bool = True,
) -> np.ndarray:
    """Channel whitening matrix ``Sigma^{-1/2}`` from within-condition residuals.

    Parameters
    ----------
    x : (n_trials, C, T).
    method : ``"ledoit-wolf"`` (shrinkage, needs sklearn), ``"diag"`` (per-channel
        variance only) or ``"none"`` (identity).
    pool_time : estimate one covariance over all timepoints (recommended at
        200 Hz with ~2880 trials); otherwise the caller must whiten per
        timepoint themselves.
    """
    x_arr = np.asarray(to_numpy(x), dtype=np.float64)
    if x_arr.ndim == 2:
        x_arr = x_arr[:, :, None]
    c = np.asarray(to_numpy(cond_ids), dtype=np.int64).ravel()
    n_ch = x_arr.shape[1]

    if method == "none":
        return np.eye(n_ch)

    # residualise each trial against its condition mean
    resid = np.empty_like(x_arr)
    for cid in np.unique(c):
        sel = np.flatnonzero(c == cid)
        resid[sel] = x_arr[sel] - x_arr[sel].mean(axis=0, keepdims=True)
    if not pool_time:
        warnings.warn(
            "estimate_whitener(pool_time=False) still returns a time-pooled "
            "whitener; per-timepoint whitening is rarely stable at n=8 repeats.",
            RuntimeWarning,
        )
    flat = np.moveaxis(resid, 1, -1).reshape(-1, n_ch)  # (n_trials*T, C)

    if method == "diag":
        var = np.maximum(flat.var(axis=0, ddof=1), 1e-20)
        return np.diag(1.0 / np.sqrt(var))

    if method != "ledoit-wolf":
        raise ValueError(f"unknown whitener method {method!r}")
    try:
        from sklearn.covariance import LedoitWolf

        cov = LedoitWolf(assume_centered=True).fit(flat).covariance_
    except Exception as exc:  # pragma: no cover - fallback path
        warnings.warn(
            f"LedoitWolf unavailable/failed ({exc}); falling back to diagonal whitener",
            RuntimeWarning,
        )
        var = np.maximum(flat.var(axis=0, ddof=1), 1e-20)
        return np.diag(1.0 / np.sqrt(var))

    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.maximum(eigval, 1e-12 * float(eigval.max()))
    return eigvec @ np.diag(eigval ** -0.5) @ eigvec.T


def crossnobis_rdm(patterns: np.ndarray, whitener: Optional[np.ndarray] = None) -> np.ndarray:
    """Cross-validated Mahalanobis RDM from fold-averaged patterns.

    Parameters
    ----------
    patterns : (n_folds, n_conditions, C) patterns for ONE timepoint/window.
    whitener : (C, C) ``Sigma^{-1/2}``; identity if None.

    Notes
    -----
    Uses the identity
    ``sum_{a != b} D^a D^{bT} = (sum_a D^a)(sum_a D^a)^T - sum_a D^a D^{aT}``
    so the cross-validated Gram matrix costs two matmuls instead of
    ``n_folds^2`` of them.  Distances are unbiased and may be negative.
    """
    p = np.asarray(patterns, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError("patterns must be (n_folds, n_conditions, C)")
    if whitener is not None:
        p = p @ np.asarray(whitener, dtype=np.float64).T
    n_folds = p.shape[0]
    if n_folds < 2:
        raise ValueError("crossnobis needs >= 2 folds")

    total = p.sum(axis=0)                       # (n_cond, C)
    gram_all = total @ total.T                  # (n_cond, n_cond)
    gram_self = np.einsum("fic,fjc->ij", p, p)  # sum_a D^a D^{aT}
    n_pairs = n_folds * (n_folds - 1)
    s = (gram_all - gram_self) / n_pairs

    diag = np.diag(s)
    rdm = diag[:, None] + diag[None, :] - 2.0 * s
    np.fill_diagonal(rdm, 0.0)
    return 0.5 * (rdm + rdm.T)


def correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    """1 - Pearson correlation across features, for (n_conditions, C) patterns."""
    p = np.asarray(patterns, dtype=np.float64)
    if p.ndim == 3:  # fold-averaged -> average folds first
        p = np.nanmean(p, axis=0)
    p = p - p.mean(axis=1, keepdims=True)
    norm = np.sqrt((p * p).sum(axis=1))
    norm = np.maximum(norm, 1e-12)
    corr = (p @ p.T) / np.outer(norm, norm)
    rdm = 1.0 - np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def time_resolved_rdms(
    x: ArrayLike,
    cond_ids: ArrayLike,
    *,
    method: str = "crossnobis",
    n_folds: int = 4,
    n_conditions: int = N_CONDITIONS,
    window: int = 1,
    step: int = 1,
    whitener: Optional[np.ndarray] = None,
    whiten_method: str = "ledoit-wolf",
    seed: int = 0,
    times: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-resolved RDMs for one subject.

    Parameters
    ----------
    x : (n_trials, C, T) epochs (primary window: 0-600 ms at 200 Hz, T=120).
    cond_ids : (n_trials,) condition index ``0..n_conditions-1``.
    window : number of consecutive timepoints concatenated into the pattern
        (``window=1`` -> 64 features; ``window=5`` -> 320 features, less noisy
        RDMs, coarser temporal resolution).
    step : stride between window centres.

    Returns
    -------
    rdms : (n_windows, P) vectorised upper triangles, float32.
    valid : (n_conditions,) boolean mask of conditions with enough trials.
    centre_times : (n_windows,) window-centre times (``times`` units, or sample
        indices when ``times`` is None).
    """
    rng = np.random.default_rng(seed)
    patterns, valid = fold_averaged_patterns(
        x, cond_ids, n_folds=n_folds, n_conditions=n_conditions, rng=rng
    )
    n_t = patterns.shape[-1]
    if valid.sum() < 3:
        raise ValueError("fewer than 3 usable conditions; cannot build an RDM")

    if method == "crossnobis" and whitener is None:
        whitener = estimate_whitener(x, cond_ids, method=whiten_method)

    idx_valid = np.flatnonzero(valid)
    n_valid = idx_valid.size
    iu = upper_tri_indices(n_valid)

    starts = list(range(0, max(n_t - window + 1, 1), step))
    out = np.empty((len(starts), iu[0].size), dtype=np.float32)
    t_arr = np.asarray(to_numpy(times)) if times is not None else np.arange(n_t)
    centres = np.array([t_arr[min(s + window // 2, n_t - 1)] for s in starts])

    for w, s in enumerate(starts):
        block = patterns[:, idx_valid, :, s : s + window]          # (F, n_valid, C, w)
        block = block.reshape(block.shape[0], n_valid, -1)          # (F, n_valid, C*w)
        if method == "crossnobis":
            whit = None
            if whitener is not None:
                # patterns were flattened as (C, window) -> C*window, so the
                # whitener has to be expanded with matching block ordering
                whit = (
                    np.asarray(whitener)
                    if window == 1
                    else _block_whitener(np.asarray(whitener), window)
                )
            rdm = crossnobis_rdm(block, whit)
        elif method == "correlation":
            rdm = correlation_rdm(block)
        else:
            raise ValueError(f"unknown RDM method {method!r}")
        out[w] = rdm[iu].astype(np.float32)

    return out, valid, centres


def _block_whitener(whitener: np.ndarray, window: int) -> np.ndarray:
    """Block-diagonal whitener for patterns flattened as (C, window) -> C*window."""
    n_ch = whitener.shape[0]
    big = np.zeros((n_ch * window, n_ch * window), dtype=np.float64)
    for i in range(n_ch):
        for j in range(n_ch):
            big[i * window : (i + 1) * window, j * window : (j + 1) * window] = (
                np.eye(window) * whitener[i, j]
            )
    return big


# --------------------------------------------------------------------------- #
# model RDMs
# --------------------------------------------------------------------------- #
def rdm_from_embeddings(emb: ArrayLike, *, metric: str = "cosine") -> np.ndarray:
    """RDM from an (n_items, D) embedding matrix."""
    e = np.asarray(to_numpy(emb), dtype=np.float64)
    if metric == "cosine":
        e = l2_normalize(e)
        rdm = 1.0 - np.clip(e @ e.T, -1.0, 1.0)
    elif metric == "correlation":
        rdm = correlation_rdm(e)
    elif metric == "euclidean":
        sq = (e * e).sum(axis=1)
        rdm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * e @ e.T, 0.0))
    else:
        raise ValueError(f"unknown metric {metric!r}")
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_from_categorical(labels: ArrayLike) -> np.ndarray:
    """0 if the two items share the label, 1 otherwise."""
    lab = np.asarray(to_numpy(labels)).ravel()
    codes = pd.Categorical(lab).codes
    rdm = (codes[:, None] != codes[None, :]).astype(np.float64)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_from_continuous(values: ArrayLike, *, standardize: bool = True) -> np.ndarray:
    """Absolute-difference RDM for a continuous attribute (valence, arousal...)."""
    v = np.asarray(to_numpy(values), dtype=np.float64).ravel()
    if standardize:
        sd = np.nanstd(v)
        v = (v - np.nanmean(v)) / (sd if sd > 1e-12 else 1.0)
    rdm = np.abs(v[:, None] - v[None, :])
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_from_features(features: ArrayLike, *, metric: str = "correlation",
                      standardize: bool = True) -> np.ndarray:
    """RDM from an (n_items, F) nuisance-feature matrix (motion energy etc.)."""
    f = np.asarray(to_numpy(features), dtype=np.float64)
    if standardize:
        mu, sd = np.nanmean(f, axis=0), np.nanstd(f, axis=0)
        f = (f - mu) / np.where(sd > 1e-12, sd, 1.0)
    if metric == "correlation":
        return correlation_rdm(f)
    return rdm_from_embeddings(f, metric=metric)


def build_model_rdms(
    cond_table: pd.DataFrame,
    *,
    cond_emb: Optional[ArrayLike] = None,
    lowlevel: Optional[ArrayLike] = None,
    categorical_cols: Sequence[str] = ("material", "touch_type", "toucher", "approaching"),
    continuous_cols: Sequence[str] = ("valence", "arousal", "threat"),
    binary_cols: Sequence[str] = ("pain",),
    orientation_col: Optional[str] = "orientation",
) -> Dict[str, np.ndarray]:
    """Assemble the standard model-RDM zoo, indexed like ``cond_table``'s rows.

    ``cond_table`` must have exactly one row per condition, ordered by
    ``condition_id`` (0..359) or by ``video_id`` (for the 90-item variant).
    Missing columns are skipped with a warning rather than raising, so the same
    call works for the 90-video and 360-condition tables.
    """
    rdms: Dict[str, np.ndarray] = {}
    n = len(cond_table)
    if cond_emb is not None:
        e = np.asarray(to_numpy(cond_emb))
        if e.shape[0] != n:
            raise ValueError(f"cond_emb has {e.shape[0]} rows, table has {n}")
        rdms["video_embedding"] = rdm_from_embeddings(e, metric="cosine")
    if lowlevel is not None:
        ll = np.asarray(to_numpy(lowlevel))
        if ll.shape[0] != n:
            raise ValueError(f"lowlevel has {ll.shape[0]} rows, table has {n}")
        rdms["lowlevel"] = rdm_from_features(ll, metric="correlation")
    for col in list(categorical_cols) + list(binary_cols):
        if col in cond_table.columns:
            rdms[f"attr_{col}"] = rdm_from_categorical(cond_table[col].to_numpy())
        else:
            warnings.warn(f"build_model_rdms: column {col!r} missing, skipped")
    for col in continuous_cols:
        if col in cond_table.columns:
            rdms[f"attr_{col}"] = rdm_from_continuous(cond_table[col].to_numpy())
        else:
            warnings.warn(f"build_model_rdms: column {col!r} missing, skipped")
    if orientation_col and orientation_col in cond_table.columns:
        rdms["orientation"] = rdm_from_categorical(cond_table[orientation_col].to_numpy())
    return rdms


# --------------------------------------------------------------------------- #
# correlations
# --------------------------------------------------------------------------- #
def spearman_rdm_corr(a: ArrayLike, b: ArrayLike) -> float:
    """Spearman correlation between two vectorised RDMs."""
    a_arr = np.asarray(to_numpy(a), dtype=np.float64).ravel()
    b_arr = np.asarray(to_numpy(b), dtype=np.float64).ravel()
    ok = np.isfinite(a_arr) & np.isfinite(b_arr)
    if ok.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(a_arr[ok], b_arr[ok]).statistic)


def partial_spearman(a: ArrayLike, b: ArrayLike, controls: ArrayLike) -> float:
    """Spearman partial correlation of ``a`` and ``b`` given ``controls``.

    Rank-transform everything, then correlate the residuals of ``a`` and ``b``
    after linear regression on the ranked controls.
    """
    a_arr = np.asarray(to_numpy(a), dtype=np.float64).ravel()
    b_arr = np.asarray(to_numpy(b), dtype=np.float64).ravel()
    c_arr = np.asarray(to_numpy(controls), dtype=np.float64)
    if c_arr.ndim == 1:
        c_arr = c_arr[:, None]
    ok = np.isfinite(a_arr) & np.isfinite(b_arr) & np.isfinite(c_arr).all(axis=1)
    if ok.sum() < 4:
        return float("nan")
    ra = stats.rankdata(a_arr[ok])[None, :]
    rb = stats.rankdata(b_arr[ok])[None, :]
    rc = np.column_stack([stats.rankdata(c_arr[ok, j]) for j in range(c_arr.shape[1])])
    res_a = _residualize_rows(_zscore_rows(ra), rc)
    res_b = _residualize_rows(_zscore_rows(rb), rc)
    num = float((res_a * res_b).sum())
    den = float(np.sqrt((res_a ** 2).sum() * (res_b ** 2).sum()))
    return num / den if den > 1e-12 else float("nan")


# --------------------------------------------------------------------------- #
# the time-course analysis with video-level permutation + clusters
# --------------------------------------------------------------------------- #
@dataclass
class RSAResult:
    """Time-resolved RSA outcome for one model RDM.

    ``r`` is always the plain Spearman correlation.  ``r_partial`` is the
    correlation after the control RDMs are removed from both sides, and is
    ``None`` for control models and when no controls were supplied.

    **The p-values and clusters refer to** ``r_partial`` **whenever it exists**,
    to ``r`` otherwise -- because the permutation null was generated for the
    same quantity that is being tested.  Reporting the plain ``r`` next to a
    partial-correlation p-value would be a mismatch, so both curves are carried
    and the report prints which one carries the inference.
    """

    model: str
    times: np.ndarray
    r: np.ndarray
    r_partial: Optional[np.ndarray]
    p_pointwise: np.ndarray
    clusters: pd.DataFrame
    null_max_cluster: np.ndarray
    n_perm: int
    controls: Tuple[str, ...] = ()
    noise_ceiling: Optional[Tuple[float, float]] = None
    #: per-timepoint (1 - alpha_pointwise) quantile of the permutation null for
    #: the SAME statistic as the inference curve; lets a caller re-test resampled
    #: curves against the identical threshold (D23 onset bootstrap).
    point_thresh: Optional[np.ndarray] = None

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "model": self.model,
                "time": self.times,
                "r": self.r,
                "p_pointwise": self.p_pointwise,
            }
        )
        if self.r_partial is not None:
            df["r_partial"] = self.r_partial
        df["controls"] = ",".join(self.controls) if self.controls else ""
        df["in_cluster"] = False
        for _, row in self.clusters.iterrows():
            m = (self.times >= row["t_start"]) & (self.times <= row["t_end"])
            df.loc[m, "in_cluster"] = True
            df.loc[m, "cluster_p"] = row["p_cluster"]
        return df


def _cluster_inference(
    obs: np.ndarray,
    null_curves: np.ndarray,
    times: np.ndarray,
    *,
    alpha_pointwise: float = 0.05,
    tail: str = "greater",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Cluster-mass inference given observed and permuted time courses.

    The pointwise threshold is the permutation quantile at each timepoint (so it
    adapts to the time-varying null width), the cluster statistic is the summed
    excess over threshold, and the correction is the max-cluster-mass null built
    from the same permutations -- i.e. the exchangeable unit of the cluster test
    is the base video, inherited from ``null_curves``.
    """
    n_perm, n_t = null_curves.shape
    if tail == "less":
        obs, null_curves = -obs, -null_curves
    thresh = np.quantile(null_curves, 1.0 - alpha_pointwise, axis=0)

    p_point = np.array(
        [
            (1 + np.sum(null_curves[:, t] >= obs[t])) / (1 + n_perm)
            for t in range(n_t)
        ]
    )

    def _clusters(curve: np.ndarray) -> List[Tuple[int, int, float]]:
        above = curve > thresh
        out: List[Tuple[int, int, float]] = []
        start = None
        for i, flag in enumerate(above):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                out.append((start, i - 1, float(np.sum(curve[start:i] - thresh[start:i]))))
                start = None
        if start is not None:
            out.append((start, n_t - 1, float(np.sum(curve[start:] - thresh[start:]))))
        return out

    null_max = np.zeros(n_perm, dtype=np.float64)
    for p in range(n_perm):
        cl = _clusters(null_curves[p])
        null_max[p] = max((c[2] for c in cl), default=0.0)

    rows = []
    for s, e, mass in _clusters(obs):
        p_c = float((1 + np.sum(null_max >= mass)) / (1 + n_perm))
        rows.append(
            {
                "t_start": float(times[s]),
                "t_end": float(times[e]),
                "i_start": int(s),
                "i_end": int(e),
                "cluster_mass": mass,
                "peak_r": float(np.max(obs[s : e + 1])),
                "p_cluster": p_c,
            }
        )
    return pd.DataFrame(rows), null_max, p_point


def rsa_time_course(
    eeg_rdms: ArrayLike,
    model_rdms: Dict[str, np.ndarray],
    *,
    times: Optional[ArrayLike] = None,
    control_models: Sequence[str] = ("lowlevel",),
    valid_conditions: Optional[ArrayLike] = None,
    video_of_condition: Optional[ArrayLike] = None,
    orientation_of_condition: Optional[ArrayLike] = None,
    n_perm: int = 1000,
    seed: int = 0,
    strata: Optional[ArrayLike] = None,
    alpha_pointwise: float = 0.05,
    n_base: int = N_BASE_VIDEOS,
) -> Dict[str, RSAResult]:
    """Spearman + partial-Spearman RSA time courses with video-level permutation.

    Parameters
    ----------
    eeg_rdms : (n_times, P) vectorised EEG RDMs, as returned by
        :func:`time_resolved_rdms`.
    model_rdms : dict name -> (n_cond, n_cond) square RDM over the SAME
        conditions (and same ordering) used for ``eeg_rdms``.
    control_models : names in ``model_rdms`` to partial out (default: the
        low-level motion-energy/luminance RDM).  Control models are themselves
        reported without partialling.
    valid_conditions : boolean mask that was applied when building ``eeg_rdms``;
        the model RDMs are subset to the same conditions.
    video_of_condition / orientation_of_condition : mapping used to lift a
        base-video permutation onto conditions.  Defaults follow the contract
        ``condition_id = (video_id-1)*4 + orientation``.
    strata : per-base-video labels for a stratified (e.g. material-matched) null.

    Returns
    -------
    dict name -> :class:`RSAResult`.
    """
    eeg = np.asarray(to_numpy(eeg_rdms), dtype=np.float64)
    if eeg.ndim == 1:
        eeg = eeg[None, :]
    n_times, n_pairs = eeg.shape
    t_axis = np.asarray(to_numpy(times)) if times is not None else np.arange(n_times)

    # ---- restrict model RDMs to the conditions present in the EEG RDM ------
    any_model = next(iter(model_rdms.values()))
    n_cond_full = np.asarray(any_model).shape[0]
    if valid_conditions is None:
        keep = np.arange(n_cond_full)
    else:
        keep = np.flatnonzero(np.asarray(to_numpy(valid_conditions), dtype=bool))
    n_keep = keep.size
    iu = upper_tri_indices(n_keep)
    if iu[0].size != n_pairs:
        raise ValueError(
            f"eeg_rdms has {n_pairs} pairs but the (masked) model RDMs imply "
            f"{iu[0].size}; check valid_conditions"
        )

    if video_of_condition is None:
        video_of_condition = np.arange(n_cond_full) // N_ORIENTATIONS
    if orientation_of_condition is None:
        orientation_of_condition = np.arange(n_cond_full) % N_ORIENTATIONS
    v_of_c = np.asarray(to_numpy(video_of_condition), dtype=np.int64)
    o_of_c = np.asarray(to_numpy(orientation_of_condition), dtype=np.int64)

    # rank the EEG side once
    eeg_ranks = np.vstack([stats.rankdata(row) for row in eeg])
    eeg_z = _zscore_rows(eeg_ranks)

    control_ranks = None
    control_names: Tuple[str, ...] = ()
    present_controls = [c for c in control_models if c in model_rdms]
    if present_controls:
        cols = []
        for name in present_controls:
            sub = np.asarray(model_rdms[name], dtype=np.float64)[np.ix_(keep, keep)]
            cols.append(stats.rankdata(sub[iu]))
        control_ranks = np.column_stack(cols)
        control_names = tuple(present_controls)

    eeg_z_res = _residualize_rows(eeg_z, control_ranks)
    eeg_z_res = _zscore_rows(eeg_z_res)

    # precompute the permutations once so every model shares them (needed for a
    # coherent max-stat correction across models downstream)
    rng = np.random.default_rng(seed)
    perms = [sample_base_permutation(n_base, rng=rng, strata=strata) for _ in range(n_perm)]
    # map base permutation -> permutation of kept condition rows
    cond_src_cache: List[np.ndarray] = []
    inv_keep = np.full(n_cond_full, -1, dtype=np.int64)
    inv_keep[keep] = np.arange(n_keep)
    n_repaired = 0
    for perm in perms:
        src_full = condition_permutation_from_base(
            perm, video_of_condition=v_of_c,
            orientation_of_condition=o_of_c, n_conditions=n_cond_full,
        )
        mapped = inv_keep[src_full[keep]]
        if np.any(mapped < 0):
            # A permuted partner fell outside the kept set. This only happens
            # when valid_conditions drops individual orientations rather than
            # whole base videos; the repair keeps the permutation a bijection by
            # filling the gaps with the unused kept indices.
            n_repaired += 1
            used = set(mapped[mapped >= 0].tolist())
            spare = [i for i in range(n_keep) if i not in used]
            holes = np.flatnonzero(mapped < 0)
            mapped[holes] = np.asarray(spare[: holes.size], dtype=np.int64)
        cond_src_cache.append(mapped)
    if n_repaired:
        warnings.warn(
            f"{n_repaired}/{n_perm} permutations needed repair because "
            "valid_conditions drops individual orientations rather than whole "
            "base videos. Drop conditions at base-video granularity to keep the "
            "null exactly exchangeable.",
            RuntimeWarning,
        )

    results: Dict[str, RSAResult] = {}
    for name, rdm in model_rdms.items():
        sub = np.asarray(rdm, dtype=np.float64)[np.ix_(keep, keep)]
        rank_mat = rank_matrix(sub)
        model_z = _zscore_rows(rank_mat[iu][None, :])[0]

        is_control = name in control_names
        if is_control or control_ranks is None:
            target_eeg = eeg_z
            model_used = model_z
            r_partial_curve = None
        else:
            target_eeg = eeg_z_res
            model_used = _zscore_rows(
                _residualize_rows(model_z[None, :], control_ranks)
            )[0]
            r_partial_curve = np.zeros(n_times)

        r_obs_plain = (eeg_z @ model_z) / n_pairs
        r_obs = (target_eeg @ model_used) / n_pairs
        if r_partial_curve is not None:
            r_partial_curve = r_obs

        null = np.empty((n_perm, n_times), dtype=np.float64)
        for i, mapped in enumerate(cond_src_cache):
            vec = rank_mat[np.ix_(mapped, mapped)][iu]
            vz = _zscore_rows(vec[None, :])
            if not is_control and control_ranks is not None:
                vz = _zscore_rows(_residualize_rows(vz, control_ranks))
            null[i] = (target_eeg @ vz[0]) / n_pairs

        clusters, null_max, p_point = _cluster_inference(
            r_obs, null, t_axis, alpha_pointwise=alpha_pointwise
        )
        point_thresh = np.quantile(null, 1.0 - alpha_pointwise, axis=0)
        results[name] = RSAResult(
            model=name,
            times=t_axis,
            r=r_obs_plain,
            r_partial=r_partial_curve,
            p_pointwise=p_point,
            clusters=clusters,
            null_max_cluster=null_max,
            n_perm=n_perm,
            controls=() if is_control else control_names,
            point_thresh=point_thresh,
        )
    return results


# --------------------------------------------------------------------------- #
# RSA noise ceiling (Nili et al. style)
# --------------------------------------------------------------------------- #
def rsa_noise_ceiling(subject_rdms: ArrayLike) -> Tuple[float, float]:
    """Upper / lower bound on the RDM correlation any model could achieve.

    Parameters
    ----------
    subject_rdms : (n_subjects, P) vectorised RDMs (one timepoint / window).

    Returns
    -------
    (lower, upper) mean Spearman correlation, leave-one-out and including-self.
    A model correlation above ``lower`` is already at the group-consistent
    ceiling; anything reported without these two numbers is uninterpretable.
    """
    r = np.asarray(to_numpy(subject_rdms), dtype=np.float64)
    if r.ndim != 2 or r.shape[0] < 3:
        return (float("nan"), float("nan"))
    ranks = np.vstack([stats.rankdata(row) for row in r])
    group = ranks.mean(axis=0)
    upper = float(np.mean([
        float(stats.spearmanr(ranks[i], group).statistic) for i in range(ranks.shape[0])
    ]))
    lower_vals = []
    for i in range(ranks.shape[0]):
        loo = np.delete(ranks, i, axis=0).mean(axis=0)
        lower_vals.append(float(stats.spearmanr(ranks[i], loo).statistic))
    return (float(np.mean(lower_vals)), upper)
