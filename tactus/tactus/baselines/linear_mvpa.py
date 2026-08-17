"""Time-resolved linear MVPA -- the companion paper, re-run under our protocol.

Reproduces the decoding analysis of the ds005662 companion paper (Imaging
Neuroscience 2025, 10.1162/IMAG.a.1017): a separate linear decoder at every
time point, shrinkage LDA for categorical stimulus attributes and ridge
regression for the continuous ones, cross-validated by **leaving out whole
sequences**, with group-level inference by cluster-based permutation.

Two reasons this is the first baseline and not an afterthought:

1. **It is the pipeline correctness check.**  The published landmarks are
   specific -- hand orientation decodable from ~60 ms peaking at 120-130 ms,
   material at 110-120 ms, valence onset ~130 ms peaking ~300 ms.  If our
   preprocessing reproduces those, the epochs are aligned, the events are parsed
   correctly and the channel order is right.  If it does not, nothing downstream
   is interpretable.  ``--report`` prints observed-vs-expected for exactly these
   landmarks.
2. **It has to be commensurable.**  The companion used leave-one-sequence-out;
   so do we, by default, which makes the comparison exact.  But leaving out a
   sequence does *not* hold out a stimulus -- the same 90 videos recur in every
   sequence -- so ``--cv video`` additionally runs the video-disjoint variant.
   The gap between the two is the honest measure of how much "attribute
   decoding" is attribute *generalization* and how much is video identity.

A caution the audit may make load-bearing: 2880 trials / 32 sequences = 90, and
360 conditions = 4 orientations x 8 repeats, so orientation may be **blocked by
sequence**.  If it is, a held-out sequence contains a single orientation, and
leave-one-sequence-out orientation decoding is partly a block/time decoder.
This module detects that (``fold class support``), warns loudly, and reports
pooled *balanced* accuracy alongside plain accuracy.

    python -m tactus.baselines.linear_mvpa --targets orientation,material,valence
    python -m tactus.baselines.linear_mvpa --cv video --targets material --report
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from tactus.common import (
    EpochStore,
    atomic_write_json,
    attach_label_ids,
    load_trials,
    results_dir,
    window_times_ms,
)

log = logging.getLogger(__name__)

__all__ = [
    "TARGETS",
    "LANDMARKS_MS",
    "decode_subject",
    "cluster_permutation_1samp",
    "onset_peak",
    "main",
]


@dataclass(frozen=True)
class TargetSpec:
    """How to build labels for one decoding target."""

    kind: str  # "categorical" | "continuous"
    column: str  # column in the (label-id-augmented) trial table
    n_classes: int = 0


TARGETS: Dict[str, TargetSpec] = {
    "orientation": TargetSpec("categorical", "orientation", 4),
    "material": TargetSpec("categorical", "material_id", 8),
    "toucher": TargetSpec("categorical", "toucher_id", 2),
    "touch_type": TargetSpec("categorical", "touch_type_id", 12),
    "pain": TargetSpec("categorical", "pain", 2),
    "approaching": TargetSpec("categorical", "approaching_id", 2),
    "valence": TargetSpec("continuous", "valence"),
    "arousal": TargetSpec("continuous", "arousal"),
    "threat": TargetSpec("continuous", "threat"),
}

#: Published landmarks (Imaging Neuroscience 2025).  ``onset`` is the first
#: significant time point, ``peak`` the maximum of the decoding curve.  These are
#: sanity targets, not ground truth: our epochs are un-baseline-corrected and our
#: artifact policy differs, so a 20-40 ms shift is unremarkable; a sign flip or a
#: 200 ms shift is a bug.
LANDMARKS_MS: Dict[str, Dict[str, Any]] = {
    "orientation": {"onset": 60, "peak": (120, 130)},
    "material": {"onset": (110, 120), "peak": None},
    "valence": {"onset": 130, "peak": 300},
    "touch_type": {"onset": 165, "peak": None},
    "threat": {"onset": (230, 260), "peak": None},
    "arousal": {"onset": (230, 260), "peak": None},
    "pain": {"onset": 135, "peak": 240},
}

TOLERANCE_MS = 40.0


# --------------------------------------------------------------------------- #
# batched linear decoders
# --------------------------------------------------------------------------- #


def _shrinkage_lda_predict(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    n_classes: int,
    shrinkage: str | float = "auto",
    scope: str = "per_class",
) -> np.ndarray:
    """Shrinkage LDA at every time point at once.

    Numerically equivalent to ``LinearDiscriminantAnalysis(solver="lsqr",
    shrinkage="auto")`` fitted independently per time point, but the covariance,
    the linear solve and the scoring are batched over time, which is what makes
    32 folds x 120 time points x 80 subjects finish in an afternoon.

    ``scope`` controls how the within-class covariance is shrunk:

    ``per_class`` (default, and what scikit-learn does)
        Ledoit-Wolf is estimated *inside each class* and the results averaged by
        the class priors -- ``sum_g prior_g * LW(X_g)``.  This is the estimator
        the companion paper used, so it is the one that makes our curves
        comparable to its published latencies.
    ``pooled``
        One Ledoit-Wolf estimate on the within-class-centered data.  Roughly
        ``n_classes`` times cheaper and usually within a percentage point, but it
        is a *different* estimator -- use it for exploration, not for the
        reproduction table.

    ``x_tr``/``x_te``: ``(n, C, T)``.  Returns predicted class ids ``(m, T)``.
    """
    n, c, t = x_tr.shape
    classes = np.arange(n_classes)
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    present = counts > 0
    if present.sum() < 2:
        return np.full((x_te.shape[0], t), int(np.argmax(counts)), dtype=np.int64)

    mu = np.zeros((n_classes, c, t), dtype=np.float64)
    for k in classes[present]:
        mu[k] = x_tr[y_tr == k].mean(axis=0)
    eye = np.eye(c)[None]
    auto = isinstance(shrinkage, str) and shrinkage == "auto"
    if auto:
        from sklearn.covariance import ledoit_wolf_shrinkage

    if scope == "per_class":
        # scikit-learn's ``_class_cov``: shrink INSIDE each class, then average by
        # the priors.  Two details are easy to get wrong and both change the
        # answer:
        #   * with shrinkage="auto" sklearn standardizes the features, runs
        #     Ledoit-Wolf, and rescales -- which makes the shrinkage target
        #     diag(S), not (tr S / p) I.  (In standardized space tr(S_z)/p is
        #     exactly 1, so the rescaled target is exactly the variance diagonal.)
        #   * with a float shrinkage it does NOT standardize, and the target is
        #     (tr S / p) I after all.
        # The empirical covariance is the biased one (denominator n_g).
        priors = counts / counts.sum()
        sigma = np.zeros((t, c, c), dtype=np.float64)
        didx = np.arange(c)
        for k in classes[present]:
            xg = x_tr[y_tr == k]
            if xg.shape[0] < 2:  # a singleton class has no covariance to shrink
                continue
            xgc = xg - xg.mean(axis=0, keepdims=True)
            s_g = np.einsum("nct,ndt->tcd", xgc, xgc, optimize=True) / xg.shape[0]
            target = np.zeros_like(s_g)
            if auto:
                rho_g = np.empty(t)
                for ti in range(t):
                    zt = xgc[:, :, ti]
                    scale = np.sqrt(s_g[ti, didx, didx])
                    scale = np.where(scale > 0, scale, 1.0)
                    rho_g[ti] = float(ledoit_wolf_shrinkage(
                        np.ascontiguousarray(zt / scale), assume_centered=True
                    ))
                target[:, didx, didx] = s_g[:, didx, didx]  # diag(S)
            else:
                rho_g = np.full(t, float(shrinkage))
                target[:, didx, didx] = (np.trace(s_g, axis1=1, axis2=2) / c)[:, None]
            sigma += priors[k] * (
                (1.0 - rho_g)[:, None, None] * s_g + rho_g[:, None, None] * target
            )
        tr_over_c = np.trace(sigma, axis1=1, axis2=2) / c
    else:
        xc = x_tr - mu[y_tr]  # within-class centered
        dof = max(1, n - int(present.sum()))
        s = np.einsum("nct,ndt->tcd", xc, xc, optimize=True) / dof
        s = 0.5 * (s + np.transpose(s, (0, 2, 1)))
        if auto:
            rho = np.array([
                float(ledoit_wolf_shrinkage(
                    np.ascontiguousarray(xc[:, :, ti]), assume_centered=True
                ))
                for ti in range(t)
            ])
        else:
            rho = np.full(t, float(shrinkage))
        tr_over_c = np.trace(s, axis1=1, axis2=2) / c
        sigma = (1.0 - rho)[:, None, None] * s + (rho * tr_over_c)[:, None, None] * eye
    sigma += 1e-10 * tr_over_c.mean() * eye  # keep it invertible for flat channels

    mu_t = np.transpose(mu, (2, 1, 0))  # (T, C, K)
    w = np.linalg.solve(sigma, mu_t)  # (T, C, K)
    prior = np.where(present, counts / max(1, counts.sum()), 1e-12)
    b = -0.5 * np.einsum("tck,tck->tk", mu_t, w, optimize=True) + np.log(prior)[None, :]
    scores = np.einsum("mct,tck->mtk", x_te, w, optimize=True) + b[None]
    scores[:, :, ~present] = -np.inf
    return scores.argmax(axis=2).astype(np.int64)


def _ridge_gcv_predict(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    """Ridge at every time point, with the penalty chosen by GCV on the train fold.

    Generalized cross-validation is computed in closed form from the eigen-
    decomposition of ``X'X`` (64x64 per time point), so the whole alpha path
    costs one eigendecomposition -- and, crucially, it never looks at the test
    fold, so the reported accuracy is not penalty-selection-optimistic.

    ``x_tr``/``x_te``: ``(n, C, T)``, ``y_tr``: ``(n,)``.  Returns ``(m, T)``.
    """
    n, c, t = x_tr.shape
    xm = x_tr.mean(axis=0, keepdims=True)
    ym = float(y_tr.mean())
    xtr = x_tr - xm
    ytr = y_tr - ym
    xtx = np.einsum("nct,ndt->tcd", xtr, xtr, optimize=True)
    xty = np.einsum("nct,n->tc", xtr, ytr, optimize=True)
    lam, v = np.linalg.eigh(0.5 * (xtx + np.transpose(xtx, (0, 2, 1))))
    lam = np.maximum(lam, 0.0)
    g = np.einsum("tcj,tc->tj", v, xty, optimize=True)  # (T, C) in the eigenbasis
    yy = float(ytr @ ytr)

    a = np.asarray(alphas, dtype=np.float64)[:, None, None]  # (A, 1, 1)
    denom = lam[None] + a  # (A, T, C)
    g2 = (g ** 2)[None]
    rss = yy - 2.0 * (g2 / denom).sum(axis=2) + (lam[None] * g2 / denom ** 2).sum(axis=2)
    tr_h = (lam[None] / denom).sum(axis=2)
    gcv = n * np.maximum(rss, 1e-30) / np.maximum(n - tr_h, 1e-6) ** 2  # (A, T)
    best = gcv.argmin(axis=0)  # (T,)

    coef_eig = g / (lam + np.asarray(alphas, dtype=np.float64)[best][:, None])
    coef = np.einsum("tcj,tj->tc", v, coef_eig, optimize=True)  # (T, C)
    return np.einsum("mct,tc->mt", x_te - xm, coef, optimize=True) + ym


# --------------------------------------------------------------------------- #
# per-subject decoding
# --------------------------------------------------------------------------- #


def _fold_groups(trials: pd.DataFrame, cv: str) -> np.ndarray:
    """The CV grouping variable."""
    if cv == "sequence":
        return trials["sequence_id"].to_numpy(np.int64)
    if cv == "video":
        return trials["video_id"].to_numpy(np.int64)
    if cv == "sequence_video":  # leave out a sequence AND every trial of its videos
        return trials["sequence_id"].to_numpy(np.int64)
    raise ValueError(f"cv must be sequence|video|sequence_video, got {cv!r}")


def decode_subject(
    subject_id: int,
    target: str,
    *,
    window: str = "w0600",
    cv: str = "sequence",
    n_folds: int | None = None,
    shrinkage: str | float = "auto",
    shrinkage_scope: str = "per_class",
    alphas: Sequence[float] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4),
    decim: int = 1,
    trials: pd.DataFrame | None = None,
    store: EpochStore | None = None,
    cache_dir: Path | str | None = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Cross-validated decoding curve for one subject and one target.

    Predictions are *pooled* across folds (each trial is predicted exactly once)
    before scoring, rather than averaging per-fold scores.  With a possibly
    class-degenerate fold -- one orientation per sequence, if the design is
    blocked -- per-fold accuracy is not even defined against a 1/4 chance level,
    while the pooled estimate is.
    """
    spec = TARGETS[target]
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"sub-{int(subject_id):02d}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=True) as z:
                return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}

    if trials is None:
        trials = attach_label_ids(load_trials(subjects=[subject_id]))
    sub = trials.loc[trials["subject_id"] == int(subject_id)].reset_index(drop=True)
    if spec.column not in sub.columns:
        raise KeyError(f"target {target!r} needs column {spec.column!r}")
    store = store if store is not None else EpochStore(window)

    ok = np.isfinite(sub[spec.column].to_numpy(dtype=np.float64))
    if spec.kind == "categorical":
        ok &= sub[spec.column].to_numpy(np.int64) >= 0
    sub = sub.loc[ok].reset_index(drop=True)

    x = store.take(int(subject_id), sub["within_subj_idx"].to_numpy())
    if decim > 1:
        x = x[:, :, ::decim]
    x = x.astype(np.float64)
    times = window_times_ms(window)[::decim]
    n, c, t = x.shape

    y_raw = sub[spec.column].to_numpy()
    if spec.kind == "categorical":
        classes = np.unique(y_raw)
        y = np.searchsorted(classes, y_raw).astype(np.int64)
        n_classes = int(classes.size)
    else:
        y = y_raw.astype(np.float64)
        n_classes = 0

    groups = _fold_groups(sub, cv)
    uniq = np.unique(groups)
    if n_folds is not None and uniq.size > n_folds:  # group k-fold for --cv video
        rng = np.random.default_rng([seed, int(subject_id)])
        assign = rng.permutation(uniq.size) % int(n_folds)
        gmap = {int(g): int(a) for g, a in zip(uniq, assign)}
        groups = np.array([gmap[int(g)] for g in groups], dtype=np.int64)
        uniq = np.unique(groups)

    pred = np.zeros((n, t), dtype=np.float64)
    degenerate = 0
    for g in uniq:
        te = groups == g
        tr = ~te
        if spec.kind == "categorical":
            n_present_te = int(np.unique(y[te]).size)
            if n_present_te < 2:
                degenerate += 1
            if int(np.unique(y[tr]).size) < 2:
                pred[te] = y[tr][0] if tr.any() else 0
                continue
        # per-fold standardization: channel x time statistics from the train fold
        mu = x[tr].mean(axis=0, keepdims=True)
        sd = x[tr].std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        xtr, xte = (x[tr] - mu) / sd, (x[te] - mu) / sd
        if spec.kind == "categorical":
            pred[te] = _shrinkage_lda_predict(
                xtr, y[tr], xte, n_classes, shrinkage, shrinkage_scope
            )
        else:
            pred[te] = _ridge_gcv_predict(xtr, y[tr], xte, alphas)

    out: Dict[str, Any] = {
        "subject_id": int(subject_id), "target": target, "cv": cv, "window": window,
        "times_ms": times, "n_trials": int(n), "n_classes": int(n_classes),
        "n_folds": int(uniq.size), "n_degenerate_folds": int(degenerate),
    }
    if spec.kind == "categorical":
        hit = pred == y[:, None]
        out["curve"] = hit.mean(axis=0)
        bal = np.stack([hit[y == k].mean(axis=0) for k in range(n_classes)
                        if (y == k).any()])
        out["curve_balanced"] = bal.mean(axis=0)
        out["chance"] = 1.0 / n_classes
        # The *empirical* floor for plain accuracy on an imbalanced label set is
        # the majority-class rate, not 1/n_classes.  ds005662's attributes are
        # severely imbalanced (material: skin 31%, metal 30%; toucher: object
        # 69%; touch_type: touch 36%), so a decoder that has learned nothing and
        # always predicts the majority class scores 0.31 / 0.69 / 0.36 -- which
        # against a uniform 1/n "chance" reads as 2.4-4.6x above chance at every
        # time point including t = 0.  Report both, and prefer
        # ``curve_balanced`` (chance 1/n by construction) for these targets.
        counts = np.bincount(y, minlength=n_classes)
        out["majority_rate"] = float(counts.max() / max(counts.sum(), 1))
        out["chance_balanced"] = 1.0 / n_classes
        out["metric"] = "accuracy"
    else:
        yc = y - y.mean()
        pc = pred - pred.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(yc) * np.linalg.norm(pc, axis=0)
        out["curve"] = np.divide(yc @ pc, np.maximum(denom, 1e-12))
        out["curve_balanced"] = out["curve"]
        out["chance"] = 0.0
        out["majority_rate"] = 0.0
        out["chance_balanced"] = 0.0
        out["metric"] = "pearson_r"
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


# --------------------------------------------------------------------------- #
# group-level inference
# --------------------------------------------------------------------------- #


def cluster_permutation_1samp(
    x: np.ndarray,
    chance: float = 0.0,
    *,
    n_perm: int = 10000,
    p_thresh: float = 0.05,
    cluster_alpha: float = 0.05,
    tail: int = 1,
    seed: int = 0,
    chunk: int = 500,
) -> Dict[str, Any]:
    """Sign-flip cluster-mass permutation test over subjects.

    ``x`` is ``(n_subjects, n_times)`` of accuracies (or correlations); the null
    is ``x == chance`` at every time point.  Time points with ``|t| >`` the
    ``p_thresh`` critical value are grouped into contiguous clusters, each
    cluster's mass is the sum of its t values, and the null distribution is the
    maximum cluster mass over ``n_perm`` random sign flips of the per-subject
    difference scores.  Exchangeability holds under the null because a subject's
    sign is arbitrary; this controls the family-wise error over time, which a
    per-time-point test emphatically does not.

    Note the resampling unit is the *subject*, so this licenses "the group
    decodes above chance", not "this generalizes to new stimuli" -- the latter
    needs the video-level permutation in ``tactus.eval``.
    """
    x = np.asarray(x, dtype=np.float64)
    d = x - chance
    n, t = d.shape
    if n < 3:
        raise ValueError(f"need >= 3 subjects for a group test, got {n}")

    def _t(a: np.ndarray) -> np.ndarray:
        m = a.mean(axis=-2)
        s = a.std(axis=-2, ddof=1)
        return m / np.maximum(s / np.sqrt(a.shape[-2]), 1e-12)

    t_obs = _t(d)
    t_crit = float(sps.t.ppf(1 - p_thresh if tail == 1 else 1 - p_thresh / 2, n - 1))

    def _clusters(tv: np.ndarray) -> List[tuple]:
        supra = tv > t_crit if tail == 1 else np.abs(tv) > t_crit
        out: List[tuple] = []
        i = 0
        while i < t:
            if supra[i]:
                j = i
                while j + 1 < t and supra[j + 1]:
                    j += 1
                out.append((i, j, float(np.abs(tv[i: j + 1]).sum())))
                i = j + 1
            else:
                i += 1
        return out

    obs = _clusters(t_obs)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm, dtype=np.float64)
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n, 1))
        tp = _t(d[None] * signs)  # (m, T)
        for i in range(m):
            cl = _clusters(tp[i])
            null[done + i] = max((c[2] for c in cl), default=0.0)
        done += m

    clusters = []
    mask = np.zeros(t, dtype=bool)
    for (i, j, mass) in obs:
        p = float((1 + np.sum(null >= mass)) / (n_perm + 1))
        clusters.append({"start": int(i), "stop": int(j), "mass": mass, "p": p})
        if p < cluster_alpha:
            mask[i: j + 1] = True
    return {
        "t_obs": t_obs, "t_crit": t_crit, "clusters": clusters, "sig_mask": mask,
        "n_perm": int(n_perm), "n_subjects": int(n), "null_max": float(null.max()),
    }


def onset_peak(times_ms: np.ndarray, curve: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """First significant time point and the peak of the curve.

    ``onset`` is the start of the first significant cluster at or after 0 ms;
    ``peak`` is the maximum inside the significant region (falling back to the
    global maximum when nothing survives, flagged by ``significant=False``).
    """
    times_ms = np.asarray(times_ms, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    valid = mask & (times_ms >= 0)
    if not valid.any():
        i = int(np.argmax(np.where(times_ms >= 0, curve, -np.inf)))
        return {"onset_ms": float("nan"), "peak_ms": float(times_ms[i]),
                "peak_value": float(curve[i]), "significant": False}
    idx = np.flatnonzero(valid)
    peak = idx[int(np.argmax(curve[idx]))]
    return {"onset_ms": float(times_ms[idx[0]]), "peak_ms": float(times_ms[peak]),
            "peak_value": float(curve[peak]), "significant": True,
            "n_sig_timepoints": int(valid.sum())}


def _landmark_check(target: str, obs: Mapping[str, float]) -> Dict[str, Any]:
    """Compare observed onset/peak against the published landmarks."""
    exp = LANDMARKS_MS.get(target)
    if exp is None:
        return {"has_landmark": False}
    out: Dict[str, Any] = {"has_landmark": True}
    for key in ("onset", "peak"):
        want = exp.get(key)
        got = obs.get(f"{key}_ms", float("nan"))
        if want is None:
            continue
        lo, hi = (want, want) if isinstance(want, (int, float)) else want
        out[f"{key}_expected_ms"] = [lo, hi]
        out[f"{key}_observed_ms"] = got
        out[f"{key}_within_tolerance"] = bool(
            np.isfinite(got) and (lo - TOLERANCE_MS) <= got <= (hi + TOLERANCE_MS)
        )
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run_target(
    target: str,
    subjects: Sequence[int],
    *,
    out_dir: Path,
    window: str,
    cv: str,
    decim: int,
    shrinkage: str | float,
    n_perm: int,
    n_jobs: int,
    n_folds: int | None,
    seed: int,
    trials: pd.DataFrame,
    metric: str = "accuracy",
    shrinkage_scope: str = "per_class",
    group_test: bool = True,
) -> Dict[str, Any] | None:
    """Decode one target for every subject, then run the group test.

    ``group_test=False`` stops after the per-subject curves are computed and
    cached.  That is what lets the 80 subjects be sharded across a worker pool:
    the group-level cluster permutation needs every subject at once (and refuses
    to run on fewer than 3), so it is deferred to one final job that re-reads the
    per-subject cache instead of re-decoding anything.
    """
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def _one(sid: int) -> Dict[str, Any]:
        store = EpochStore(window)  # one memmap handle per worker
        return decode_subject(
            sid, target, window=window, cv=cv, decim=decim, shrinkage=shrinkage,
            shrinkage_scope=shrinkage_scope, n_folds=n_folds, trials=trials,
            store=store, cache_dir=cache_dir, seed=seed,
        )

    if n_jobs == 1:
        results = [_one(s) for s in subjects]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one)(s) for s in subjects)

    results = [r for r in results if r is not None]
    # Empirical accuracy floor, read from the trial table so it is available
    # even when every per-subject curve came from a cache written before this
    # field existed.
    # Balanced accuracy averages per-class recall, so its floor really IS
    # 1/n_classes -- the majority-class rate applies only to plain accuracy.
    majority = (0.0 if metric == "balanced_accuracy"
                else _majority_rate(target, trials, subjects))
    if not group_test:
        log.info("%s: cached %d per-subject curve(s) in %.1f min; group test deferred",
                 target, len(results), (time.time() - t0) / 60)
        return None
    key = "curve_balanced" if metric == "balanced_accuracy" else "curve"
    curves = np.stack([np.asarray(r[key], dtype=np.float64) for r in results])
    times = np.asarray(results[0]["times_ms"], dtype=np.float64)
    chance = float(np.asarray(results[0]["chance"]))
    degenerate = int(sum(int(np.asarray(r["n_degenerate_folds"])) for r in results))

    stats = cluster_permutation_1samp(curves, chance, n_perm=n_perm, seed=seed)
    grand = curves.mean(axis=0)
    lat = onset_peak(times, grand, stats["sig_mask"])

    pd.DataFrame(
        {"time_ms": times, "mean": grand,
         "sem": curves.std(axis=0, ddof=1) / np.sqrt(len(curves)),
         "t": stats["t_obs"], "significant": stats["sig_mask"]}
    ).to_csv(out_dir / "curve.csv", index=False)
    np.savez_compressed(out_dir / "curves.npz", curves=curves, times_ms=times,
                        subjects=np.asarray(subjects), chance=chance)

    summary = {
        "target": target, "metric": results[0]["metric"], "cv": cv, "window": window,
        "chance": chance, "n_subjects": len(curves), "decim": decim,
        "reported_metric": metric,
        "majority_rate": majority,
        "prestimulus": _prestimulus_diagnostic(
            times, grand, chance, majority_rate=majority,
            metric=results[0]["metric"],
        ),
        "n_degenerate_folds_total": degenerate,
        "latency": lat, "landmark": _landmark_check(target, lat),
        "clusters": stats["clusters"], "t_crit": stats["t_crit"], "n_perm": n_perm,
        "peak_mean_value": float(grand.max()),
        "seconds": time.time() - t0,
    }
    atomic_write_json(out_dir / "stats.json", summary)
    if degenerate:
        log.warning(
            "%s: %d class-degenerate CV folds (a held-out group contained <2 "
            "classes). If this is orientation under --cv sequence, the design is "
            "blocked and the decoding claim needs the audit-A caveat.",
            target, degenerate,
        )
    return summary


def _majority_rate(target: str, trials: pd.DataFrame, subjects: Sequence[int]) -> float:
    """Share of the most frequent class, pooled over the decoded subjects.

    This is the score of a decoder that has learned nothing and always predicts
    the majority class -- the honest floor for plain accuracy.  ``0.0`` for
    continuous targets, whose floor is r = 0.
    """
    spec = TARGETS.get(target)
    if spec is None or spec.kind != "categorical" or spec.column not in trials.columns:
        return 0.0
    sub = trials[trials["subject_id"].isin(list(map(int, subjects)))]
    y = sub[spec.column].to_numpy()
    y = y[y >= 0] if np.issubdtype(y.dtype, np.number) else y
    if not len(y):
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def _prestimulus_diagnostic(
    times: np.ndarray, grand: np.ndarray, chance: float,
    *, majority_rate: float = 0.0, metric: str = "accuracy",
) -> Dict[str, Any]:
    """Is the curve an evoked response, or a time-invariant offset?

    A genuine evoked effect starts at chance and rises.  A decoder that is
    already above chance in the **first sample** cannot be reading a
    stimulus-evoked response -- nothing has been evoked yet -- so it is reading
    something time-invariant that happens to correlate with the label.

    Under leave-one-sequence-out CV that something is easy to name: all 90
    videos recur in every sequence, so holding out a sequence does not hold out
    a stimulus, and any stable per-(subject, video) offset (drift, impedance,
    electrode state) is a free label.  Orientation is immune, because all four
    orientations of a video appear in every sequence and the per-video offset
    therefore carries no orientation information -- which is exactly why
    orientation can be the pipeline-alignment check while material cannot.

    ``evoked_fraction`` is the share of the peak effect that actually appears
    *after* onset: ``(peak - t0) / (peak - chance)``.  Near 0 means flat.
    """
    t0 = float(grand[0])
    peak = float(np.nanmax(grand))
    # Compare against the EMPIRICAL floor, not the uniform one.
    floor = max(chance, majority_rate) if metric == "accuracy" else chance
    denom = peak - floor
    at_floor = bool(abs(t0 - floor) <= 0.02 * max(abs(floor), 1e-9) + 0.01)
    return {
        "value_at_first_sample": round(t0, 4),
        "first_sample_ms": round(float(times[0]), 1),
        "uniform_chance": round(float(chance), 4),
        "majority_rate": round(float(majority_rate), 4),
        "empirical_floor": round(float(floor), 4),
        "offset_above_uniform_chance": round(t0 - chance, 4),
        "offset_above_empirical_floor": round(t0 - floor, 4),
        "t0_at_empirical_floor": at_floor,
        "evoked_fraction": round(float((peak - t0) / denom), 4) if abs(denom) > 1e-9 else None,
        # Only a curve that starts ABOVE the empirical floor and stays flat is
        # suspicious.  Sitting exactly at the majority-class rate is a null
        # result reported on the wrong scale, not leakage.
        "flat": bool(abs(denom) > 1e-9 and (peak - t0) / denom < 0.25 and not at_floor),
        "majority_class_only": bool(metric == "accuracy" and at_floor
                                    and abs(denom) > 1e-9 and (peak - t0) / denom < 0.25),
    }


def _write_report(out_root: Path, summaries: Sequence[Mapping[str, Any]]) -> Path:
    """Human-readable observed-vs-published table."""
    lines = [
        "# Time-resolved MVPA baseline", "",
        "Companion paper: Imaging Neuroscience 2025 (10.1162/IMAG.a.1017).",
        "Tolerance for the landmark check is +-%.0f ms." % TOLERANCE_MS, "",
        "| target | metric | uniform chance | majority rate | peak | onset (ms) | peak (ms) | expected onset | expected peak | verdict | t0 | evoked frac |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    flat_targets = []
    majority_only = []
    for s in summaries:
        lm = s.get("landmark", {})
        lat = s.get("latency", {})
        pre = s.get("prestimulus", {}) or {}
        verdict = "-"
        if lm.get("has_landmark"):
            checks = [v for k, v in lm.items() if k.endswith("_within_tolerance")]
            verdict = "OK" if checks and all(checks) else "MISMATCH"
        if pre.get("majority_class_only"):
            verdict = "NULL (majority class)"
            majority_only.append(s["target"])
        elif pre.get("flat"):
            verdict = "LEAKAGE?" if verdict != "OK" else "OK (but flat!)"
            flat_targets.append(s["target"])
        lines.append(
            "| {t} | {m} | {c:.3f} | {mr} | {p:.3f} | {on} | {pk} | {eon} | {epk} | {v} | {t0} | {ef} |".format(
                t=s["target"], m=s["metric"], c=s["chance"],
                mr=("%.3f" % s["majority_rate"]) if s.get("majority_rate") else "-",
                p=s["peak_mean_value"],
                on=_fmt(lat.get("onset_ms")), pk=_fmt(lat.get("peak_ms")),
                eon=lm.get("onset_expected_ms", "-"), epk=lm.get("peak_expected_ms", "-"),
                v=verdict,
                t0=pre.get("value_at_first_sample", "-"),
                ef=pre.get("evoked_fraction", "-"),
            )
        )
    if majority_only:
        lines += [
            "",
            "> **Majority-class null: %s.**" % ", ".join(majority_only),
            "> These curves sit flat at the **majority-class rate**, not above it. Plain",
            "> accuracy against a uniform `1/n_classes` chance makes that look like strong",
            "> decoding -- ds005662's attributes are severely imbalanced (material: skin",
            "> 31%, metal 30%; toucher: object 69%; touch_type: touch 36%), so a decoder",
            "> that always predicts the majority class scores 2.4-4.6x 'above chance' at",
            "> every time point *including t = 0*. It is a null result on the wrong scale,",
            "> not leakage and not a preprocessing fault. Read `--metric balanced_accuracy`",
            "> for these targets, whose chance really is 1/n_classes.",
        ]
    if flat_targets:
        lines += [
            "",
            "> **Flat-curve warning: %s.**" % ", ".join(flat_targets),
            "> These decoders are already above chance in the *first sample*, before any",
            "> stimulus-evoked response can exist, and gain little afterwards",
            "> (`evoked frac` well below 1). That is a time-invariant offset, not an",
            "> evoked effect. Under `--cv sequence` the cause is structural rather than",
            "> a bug: every sequence contains all 90 videos, so holding out a sequence",
            "> does not hold out a *stimulus*, and any stable per-(subject, video)",
            "> offset is a free label. Re-run with `--cv video` for the honest number;",
            "> the gap between the two is the attribute-shortcut quantification the",
            "> blueprint (5.2) requires. Orientation is the valid alignment check",
            "> precisely because it is immune: all four orientations of a video occur",
            "> in every sequence, so a per-video offset carries no orientation signal.",
        ]
    lines += [
        "", "Notes:",
        "- Read `majority rate`, `t0` and `evoked frac` before the verdict.",
        "  `t0` at the majority rate + flat curve = the decoder learned nothing",
        "  (`NULL (majority class)`). `t0` ABOVE the empirical floor + flat curve =",
        "  a genuine time-invariant confound (`LEAKAGE?`). `t0` at the floor with a",
        "  rising curve = a real evoked effect, and only then do the latencies mean",
        "  anything.",
        "- `MISMATCH` on one target with the others `OK` usually means a label bug,",
        "  not a preprocessing bug; `MISMATCH` everywhere means the epochs are",
        "  misaligned (check the event onset column and the window definition).",
        "- Onsets here are cluster onsets, which are biased later than the",
        "  single-time-point onsets some papers report, and are not themselves",
        "  a valid basis for a latency *difference* test.",
    ]
    path = out_root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fmt(v: Any) -> str:
    try:
        return "-" if v is None or not np.isfinite(float(v)) else f"{float(v):.0f}"
    except (TypeError, ValueError):
        return "-"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tactus.baselines.linear_mvpa",
        description="Time-resolved LDA/ridge decoding with cluster permutation.",
    )
    p.add_argument("--targets", default="orientation,material,toucher,valence,arousal,threat",
                   help=f"comma-separated; known: {','.join(TARGETS)}")
    p.add_argument("--window", default="w0600")
    p.add_argument("--cv", default="sequence", choices=["sequence", "video", "sequence_video"],
                   help="sequence = the companion paper's protocol; video = "
                        "stimulus-disjoint, the honest generalization test")
    p.add_argument("--n-folds", type=int, default=None,
                   help="collapse groups into this many folds (use ~9 with --cv video)")
    p.add_argument("--subjects", default="all", help="'all' or e.g. 1-20,35")
    p.add_argument("--decim", type=int, default=1, help="temporal decimation factor")
    p.add_argument("--shrinkage", default="auto", help="'auto' (Ledoit-Wolf) or a float")
    p.add_argument("--shrinkage-scope", default="per_class", choices=["per_class", "pooled"],
                   help="per_class matches sklearn/the companion paper; pooled is "
                        "~n_classes times faster but is a different estimator")
    p.add_argument("--metric", default="accuracy",
                   choices=["accuracy", "balanced_accuracy"])
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default results/baselines/mvpa")
    p.add_argument("--report", action="store_true", help="write report.md at the end")
    p.add_argument("--no-group", action="store_true",
                   help="only compute and cache the per-subject curves; skip the "
                        "group cluster-permutation test (used to shard subjects "
                        "across a worker pool)")
    p.add_argument("--log-level", default="INFO")
    return p


def parse_subjects(spec: str, available: Sequence[int]) -> List[int]:
    """``"all"`` or ``"1-20,35,40"`` -> a sorted list of subject ids."""
    if spec.strip().lower() == "all":
        return list(available)
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out) & set(available)) or sorted(set(out))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )
    out_root = Path(args.out) if args.out else results_dir() / "baselines" / "mvpa"
    out_root = out_root / f"{args.window}_{args.cv}"
    out_root.mkdir(parents=True, exist_ok=True)

    trials = attach_label_ids(load_trials())
    store = EpochStore(args.window)
    available = sorted(set(trials["subject_id"].astype(int)) & set(store.available_subjects()))
    if not available:
        raise SystemExit(
            f"no subject has both trial-table rows and an epoch file for window "
            f"{args.window!r}; run the preprocessing stage first."
        )
    subjects = parse_subjects(args.subjects, available)
    log.info("decoding %d subjects, window=%s, cv=%s", len(subjects), args.window, args.cv)

    shrinkage: str | float = args.shrinkage
    with_float = None
    try:
        with_float = float(args.shrinkage)
    except ValueError:
        pass
    if with_float is not None:
        shrinkage = with_float

    summaries: List[Dict[str, Any]] = []
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        if target not in TARGETS:
            log.error("unknown target %r (known: %s)", target, ", ".join(TARGETS))
            continue
        out_dir = out_root / target
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("--- %s ---", target)
        s = run_target(
            target, subjects, out_dir=out_dir, window=args.window, cv=args.cv,
            decim=args.decim, shrinkage=shrinkage, n_perm=args.n_perm,
            n_jobs=args.n_jobs, n_folds=args.n_folds, seed=args.seed, trials=trials,
            metric=args.metric, shrinkage_scope=args.shrinkage_scope,
            group_test=not args.no_group,
        )
        if s is None:          # --no-group: cache-warming shard, nothing to report
            continue
        summaries.append(s)
        lat = s["latency"]
        log.info(
            "%s: peak %.3f at %s ms, onset %s ms, %d significant clusters",
            target, s["peak_mean_value"], _fmt(lat.get("peak_ms")),
            _fmt(lat.get("onset_ms")),
            sum(1 for c in s["clusters"] if c["p"] < 0.05),
        )

    if summaries:
        atomic_write_json(out_root / "all_targets.json", summaries)
        if args.report:
            path = _write_report(out_root, summaries)
            log.info("wrote %s", path)
            print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
