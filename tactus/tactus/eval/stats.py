"""Inference-target-aware statistics for TACTUS.

The single discipline this module enforces (BLUEPRINT v2 §5.2): **every
comparison must declare what it generalises over before a test is chosen.**

* *Fixed-stimulus* claims (model ranking, ablations, "encoder A beats encoder B
  on these 90 videos") generalise over subjects only.  Test: subject-level
  paired Wilcoxon on the same folds, n = 80.  Resolution: ~1 accuracy point.
* *Stimulus-generalising* claims ("this holds for touch videos in general")
  generalise over subjects **and** videos.  Test: crossed subject x video mixed
  model.  Resolution: ~8 accuracy points at 90 video units and p ~ 0.15.

Anything smaller than the relevant minimal detectable difference is below the
design's resolution and must be reported as "not resolvable", never as a
negative result and never as a win.  :func:`print_mdd_table` prints both
resolutions so that no ablation table gets over-interpreted.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "bootstrap_ci",
    "paired_wilcoxon",
    "crossed_mixed_model",
    "by_video_paired_test",
    "benjamini_hochberg",
    "hierarchical_fdr",
    "cluster_permutation_1d",
    "cluster_permutation_from_null",
    "mdd_paired_proportions",
    "mdd_from_sd",
    "mdd_table",
    "print_mdd_table",
    "flag_below_mdd",
    "recommend_test",
    "binomial_ci",
]

ArrayLike = Any


# --------------------------------------------------------------------------- #
# basic interval estimation
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: ArrayLike,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 0,
    method: str = "percentile",
) -> Dict[str, float]:
    """Bootstrap CI over the *unit of inference* (rows of ``values``).

    Pass subject-level scores for fixed-stimulus claims and video-level scores
    for stimulus-generalising claims.  Bootstrapping trials is never correct
    here: trials are clustered within video and within subject.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "se": np.nan}
    rng = np.random.default_rng(seed)
    est = float(statistic(v))
    if n == 1:
        return {"estimate": est, "lo": np.nan, "hi": np.nan, "n": 1, "se": np.nan}

    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(v[row]) for row in idx], dtype=np.float64)
    alpha = (100.0 - ci) / 2.0
    if method == "percentile":
        lo, hi = np.percentile(boot, [alpha, 100.0 - alpha])
    elif method == "bc":  # bias-corrected (no acceleration term)
        z0 = stats.norm.ppf(np.mean(boot < est)) if 0 < np.mean(boot < est) < 1 else 0.0
        zl, zh = stats.norm.ppf(alpha / 100.0), stats.norm.ppf(1 - alpha / 100.0)
        pl = stats.norm.cdf(2 * z0 + zl) * 100.0
        ph = stats.norm.cdf(2 * z0 + zh) * 100.0
        lo, hi = np.percentile(boot, [pl, ph])
    else:
        raise ValueError(f"unknown bootstrap method {method!r}")
    return {
        "estimate": est,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(np.std(boot, ddof=1)),
        "n": int(n),
    }


def binomial_ci(k: int, n: int, ci: float = 95.0) -> Tuple[float, float]:
    """Clopper-Pearson exact interval.

    Used for the effect-size calibration in BLUEPRINT v2 §5.2: a single fold of
    a single subject over 18 video units has a 95% interval of roughly 0-19% at
    chance, i.e. it cannot distinguish 15% from chance.  Every claim must live
    on the fold x subject aggregate.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    alpha = (100.0 - ci) / 100.0
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# fixed-stimulus inference: subject-level paired test
# --------------------------------------------------------------------------- #
def paired_wilcoxon(
    a: ArrayLike,
    b: ArrayLike,
    *,
    labels: Tuple[str, str] = ("model_a", "model_b"),
    alternative: str = "two-sided",
    n_boot: int = 10_000,
    seed: int = 0,
    unit: str = "subject",
) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test over units of inference.

    ``a`` and ``b`` must be aligned per unit (subject) and must come from the
    **same folds**; pairing across different fold assignments silently inflates
    the variance and is the most common way this test is misused.

    Returns the statistic, the exact/normal p-value, the matched-pairs
    rank-biserial correlation as effect size, the bootstrap CI of the median
    difference, and the number of usable pairs.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired_wilcoxon needs aligned, equal-length inputs")
    ok = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr, b_arr = a_arr[ok], b_arr[ok]
    d = a_arr - b_arr
    n = d.size
    if n < 3:
        return {
            "test": "wilcoxon", "unit": unit, "n_pairs": int(n),
            "statistic": np.nan, "p_value": np.nan, "effect_size_rb": np.nan,
            "median_diff": float(np.median(d)) if n else np.nan,
            "ci_lo": np.nan, "ci_hi": np.nan, "labels": labels,
            "note": "fewer than 3 usable pairs",
        }

    nz = d[d != 0]
    if nz.size == 0:
        stat, p = np.nan, 1.0
        rb = 0.0
    else:
        res = stats.wilcoxon(a_arr, b_arr, alternative=alternative, zero_method="wilcox")
        stat, p = float(res.statistic), float(res.pvalue)
        ranks = stats.rankdata(np.abs(nz))
        t_pos = float(ranks[nz > 0].sum())
        t_neg = float(ranks[nz < 0].sum())
        rb = (t_pos - t_neg) / (t_pos + t_neg) if (t_pos + t_neg) > 0 else 0.0

    boot = bootstrap_ci(d, statistic=np.median, n_boot=n_boot, seed=seed)
    return {
        "test": "wilcoxon_signed_rank",
        "unit": unit,
        "inference_target": "fixed stimuli (generalises over subjects only)",
        "n_pairs": int(n),
        "statistic": stat,
        "p_value": float(p),
        "alternative": alternative,
        "effect_size_rb": float(rb),
        "mean_diff": float(np.mean(d)),
        "median_diff": float(np.median(d)),
        "ci_lo": boot["lo"],
        "ci_hi": boot["hi"],
        "n_zero_diffs": int((d == 0).sum()),
        "labels": labels,
    }


# --------------------------------------------------------------------------- #
# stimulus-generalising inference: crossed subject x video model
# --------------------------------------------------------------------------- #
def crossed_mixed_model(
    df: pd.DataFrame,
    *,
    outcome: str = "correct",
    fixed_effects: str = "model",
    subject_col: str = "subject_id",
    video_col: str = "video_id",
    extra_random_slopes: Optional[Sequence[str]] = None,
    method: str = "lmm",
    maxiter: int = 200,
) -> Dict[str, Any]:
    """Crossed random effects for subject and video (stimulus-generalising claims).

    Fits ``outcome ~ fixed_effects + (1 | subject) + (1 | video)`` using
    statsmodels' variance-component parameterisation (a single dummy group with
    two variance components is the standard way to express crossed effects in
    statsmodels; it is slower than lme4 but gives the same target).

    ``method="lmm"`` treats the outcome as continuous (use per-(subject, video)
    accuracies, which are averages of 32 trials and are well behaved).
    ``method="glmm_binomial"`` is accepted but currently falls back to the LMM
    on the logit of the aggregated accuracy, with a warning: statsmodels has no
    crossed-random-effects binomial GLMM, and a trial-level Bernoulli fit with
    crossed effects will not converge at this size.

    Returns a dict with the coefficient table, the two variance components, the
    convergence flag, and an explicit ``inference_target`` string.
    """
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "crossed_mixed_model needs statsmodels: pip install statsmodels"
        ) from exc

    data = df.copy()
    for col in (outcome, subject_col, video_col):
        if col not in data.columns:
            raise KeyError(f"missing column {col!r}")
    data = data.dropna(subset=[outcome])

    if method == "glmm_binomial":
        warnings.warn(
            "no crossed-random-effects binomial GLMM is available in statsmodels; "
            "fitting the LMM on logit(accuracy) instead and reporting it as such",
            RuntimeWarning,
        )
        eps = 1e-3
        p = np.clip(data[outcome].to_numpy(dtype=float), eps, 1 - eps)
        data["_outcome"] = np.log(p / (1 - p))
        outcome_used = "_outcome"
        link = "logit(accuracy), LMM"
    elif method == "lmm":
        data["_outcome"] = data[outcome].to_numpy(dtype=float)
        outcome_used = "_outcome"
        link = "identity, LMM"
    else:
        raise ValueError(f"unknown method {method!r}")

    data["_grp"] = 1
    vc = {
        "subject": f"0 + C({subject_col})",
        "video": f"0 + C({video_col})",
    }
    if extra_random_slopes:
        for i, col in enumerate(extra_random_slopes):
            vc[f"extra{i}"] = f"0 + C({col})"

    formula = f"{outcome_used} ~ {fixed_effects}"
    model = smf.mixedlm(formula, data, groups=data["_grp"], vc_formula=vc, re_formula="0")
    try:
        fit = model.fit(reml=True, method="lbfgs", maxiter=maxiter)
        converged = bool(getattr(fit, "converged", True))
    except Exception as exc:  # pragma: no cover
        return {
            "test": "crossed_mixed_model",
            "error": str(exc),
            "converged": False,
            "inference_target": "stimulus-generalising (subjects x videos)",
        }

    coefs = pd.DataFrame(
        {
            "term": fit.params.index,
            "estimate": fit.params.to_numpy(),
            "se": fit.bse.reindex(fit.params.index).to_numpy(),
            "z": fit.tvalues.reindex(fit.params.index).to_numpy(),
            "p_value": fit.pvalues.reindex(fit.params.index).to_numpy(),
        }
    )
    if not converged:
        warnings.warn(
            "crossed_mixed_model did not converge; report the by-video paired "
            "test as the primary stimulus-generalising analysis instead",
            RuntimeWarning,
        )

    try:
        vcomp_vals = np.atleast_1d(np.asarray(fit.vcomp, dtype=float))
        variance_components = {
            name: float(val) for name, val in zip(vc.keys(), vcomp_vals)
        }
    except Exception:
        variance_components = {}

    return {
        "test": "crossed_mixed_model",
        "inference_target": "stimulus-generalising (crossed subjects x videos)",
        "link": link,
        "formula": f"{formula} + (1|{subject_col}) + (1|{video_col})",
        "coefficients": coefs,
        "variance_components": variance_components,
        "scale": float(getattr(fit, "scale", np.nan)),
        "converged": converged,
        "n_obs": int(len(data)),
        "n_subjects": int(data[subject_col].nunique()),
        "n_videos": int(data[video_col].nunique()),
        "summary_text": str(fit.summary()),
    }


def by_video_paired_test(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    model_col: str = "model",
    video_col: str = "video_id",
    models: Optional[Tuple[str, str]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Video-level paired Wilcoxon: the mixed model's assumption-light stand-in.

    Averages over subjects within video first, then pairs by video.  Weaker than
    the crossed model (it ignores subject-level variance) but it converges, and
    at n = 90 videos it has the same resolution ceiling, so it is the sensible
    fallback whenever the mixed model fails.
    """
    piv = (
        df.groupby([video_col, model_col])[value_col].mean().unstack(model_col)
    )
    if models is None:
        if piv.shape[1] != 2:
            raise ValueError(
                f"expected exactly 2 models, found {list(piv.columns)}; pass models=..."
            )
        models = tuple(piv.columns[:2])  # type: ignore[assignment]
    a, b = piv[models[0]].to_numpy(), piv[models[1]].to_numpy()
    res = paired_wilcoxon(a, b, labels=models, unit="video", seed=seed)
    res["inference_target"] = (
        "stimulus-generalising (videos as units; subject variance ignored)"
    )
    res["n_videos"] = int(piv.shape[0])
    return res


# --------------------------------------------------------------------------- #
# multiplicity
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals: ArrayLike, q: float = 0.05) -> Dict[str, np.ndarray]:
    """Standard BH step-up.  Returns rejections and BH-adjusted p-values."""
    p = np.asarray(pvals, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return {"reject": np.zeros(0, dtype=bool), "p_adj": np.zeros(0)}
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    p_adj = np.empty(n, dtype=np.float64)
    p_adj[order] = adj
    return {"reject": p_adj <= q, "p_adj": p_adj}


def _simes(p: np.ndarray) -> float:
    """Simes combination p-value for one family."""
    p = np.sort(np.asarray(p, dtype=np.float64).ravel())
    m = p.size
    if m == 0:
        return float("nan")
    return float(np.min(m * p / (np.arange(m) + 1)))


def hierarchical_fdr(
    df: pd.DataFrame,
    *,
    p_col: str = "p_value",
    family_col: str = "family",
    endpoint_col: str = "endpoint",
    q: float = 0.05,
) -> pd.DataFrame:
    """Benjamini-Bogomolov two-level FDR across the endpoint zoo.

    Level 1: each family (e.g. "primary", "retrieval ladder", "RSA time course",
    "confound battery", "phenotype") is summarised by its Simes p-value and BH
    is applied across families at level ``q``.
    Level 2: within each *selected* family, BH is applied at the reduced level
    ``q * R / F`` where ``R`` is the number of selected families and ``F`` the
    total.  This controls the average FDR over selected families, which is the
    right guarantee when a pre-registered primary endpoint sits alongside a
    large exploratory zoo.

    Returns the input frame with ``p_family``, ``family_selected``,
    ``q_within``, ``p_adj_within`` and ``reject`` columns added.
    """
    out = df.copy()
    for col in (p_col, family_col, endpoint_col):
        if col not in out.columns:
            raise KeyError(f"missing column {col!r}")

    fam_p = out.groupby(family_col)[p_col].apply(lambda s: _simes(s.to_numpy()))
    fam_names = list(fam_p.index)
    fam_bh = benjamini_hochberg(fam_p.to_numpy(), q=q)
    selected = dict(zip(fam_names, fam_bh["reject"]))
    n_sel = int(sum(selected.values()))
    n_fam = len(fam_names)
    q_within = q * (n_sel / n_fam) if n_fam else q

    out["p_family"] = out[family_col].map(fam_p.to_dict())
    out["family_selected"] = out[family_col].map(selected)
    out["q_within"] = np.where(out["family_selected"], q_within, np.nan)

    out["p_adj_within"] = np.nan
    out["reject"] = False
    for fam in fam_names:
        sel = out[family_col] == fam
        if not selected[fam]:
            continue
        bh = benjamini_hochberg(out.loc[sel, p_col].to_numpy(), q=q_within)
        out.loc[sel, "p_adj_within"] = bh["p_adj"]
        out.loc[sel, "reject"] = bh["p_adj"] <= q_within
    out.attrs["n_families"] = n_fam
    out.attrs["n_families_selected"] = n_sel
    out.attrs["q_level1"] = q
    out.attrs["q_level2"] = q_within
    return out


# --------------------------------------------------------------------------- #
# cluster-based permutation over time
# --------------------------------------------------------------------------- #
def cluster_permutation_1d(
    x: ArrayLike,
    *,
    times: Optional[ArrayLike] = None,
    threshold: Optional[float] = None,
    alpha_pointwise: float = 0.05,
    n_perm: int = 5000,
    tail: str = "two-sided",
    seed: int = 0,
    unit: str = "subject",
) -> Dict[str, Any]:
    """One-sample cluster-mass permutation over time by sign flipping.

    Parameters
    ----------
    x : (n_units, n_times) paired differences (or values tested against 0).
        ``n_units`` must be the *exchangeable* unit -- subjects for
        fixed-stimulus claims, videos for stimulus-generalising claims.  The
        chosen unit is recorded in the output and printed in the report; a
        cluster test run over trials is meaningless here because trials are
        clustered within video.

    Returns
    -------
    dict with the observed t-curve, the cluster table (onset, offset, mass,
    p-value), the max-cluster null, and the threshold used.
    """
    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("x must be (n_units, n_times)")
    n_units, n_t = data.shape
    t_axis = np.asarray(times, dtype=np.float64) if times is not None else np.arange(n_t, dtype=float)
    rng = np.random.default_rng(seed)

    def _tstat(d: np.ndarray) -> np.ndarray:
        m = d.mean(axis=0)
        s = d.std(axis=0, ddof=1)
        return m / np.maximum(s / math.sqrt(d.shape[0]), 1e-12)

    if threshold is None:
        df_ = n_units - 1
        crit = stats.t.ppf(
            1 - alpha_pointwise / (2 if tail == "two-sided" else 1), df_
        )
        threshold = float(crit)

    def _clusters(tvals: np.ndarray) -> List[Tuple[int, int, float]]:
        if tail == "greater":
            above = tvals > threshold
            signed = tvals
        elif tail == "less":
            above = tvals < -threshold
            signed = -tvals
        else:
            above = np.abs(tvals) > threshold
            signed = np.abs(tvals)
        out: List[Tuple[int, int, float]] = []
        start = None
        for i, flag in enumerate(above):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                out.append((start, i - 1, float(signed[start:i].sum())))
                start = None
        if start is not None:
            out.append((start, n_t - 1, float(signed[start:].sum())))
        return out

    t_obs = _tstat(data)
    obs_clusters = _clusters(t_obs)

    null_max = np.zeros(n_perm, dtype=np.float64)
    for i in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=(n_units, 1))
        cl = _clusters(_tstat(data * flips))
        null_max[i] = max((c[2] for c in cl), default=0.0)

    rows = []
    for s, e, mass in obs_clusters:
        p = float((1 + np.sum(null_max >= mass)) / (1 + n_perm))
        rows.append(
            {
                "t_start": float(t_axis[s]),
                "t_end": float(t_axis[e]),
                "i_start": int(s),
                "i_end": int(e),
                "cluster_mass": mass,
                "peak_t": float(t_obs[s : e + 1][np.argmax(np.abs(t_obs[s : e + 1]))]),
                "p_cluster": p,
                "significant": p <= 0.05,
            }
        )
    return {
        "t_curve": t_obs,
        "times": t_axis,
        "clusters": pd.DataFrame(rows),
        "null_max_cluster": null_max,
        "threshold": threshold,
        "n_units": n_units,
        "exchangeable_unit": unit,
        "tail": tail,
        "n_perm": n_perm,
    }


def cluster_permutation_from_null(
    observed: ArrayLike,
    null_curves: ArrayLike,
    *,
    times: Optional[ArrayLike] = None,
    alpha_pointwise: float = 0.05,
) -> Dict[str, Any]:
    """Cluster inference when the null already comes from video permutations.

    Use this for RSA / retrieval time courses whose null was generated by
    shuffling base videos: the exchangeable unit is already correct and the sign
    flip of :func:`cluster_permutation_1d` would be the wrong resampling scheme.
    """
    from .rsa import _cluster_inference  # single implementation, reused

    obs = np.asarray(observed, dtype=np.float64).ravel()
    null = np.asarray(null_curves, dtype=np.float64)
    t_axis = np.asarray(times, dtype=np.float64) if times is not None else np.arange(obs.size, dtype=float)
    clusters, null_max, p_point = _cluster_inference(
        obs, null, t_axis, alpha_pointwise=alpha_pointwise
    )
    return {
        "clusters": clusters,
        "null_max_cluster": null_max,
        "p_pointwise": p_point,
        "times": t_axis,
        "exchangeable_unit": "base_video",
        "n_perm": int(null.shape[0]),
    }


# --------------------------------------------------------------------------- #
# minimal detectable difference
# --------------------------------------------------------------------------- #
def mdd_from_sd(
    sd_unit: float,
    n_units: int,
    *,
    rho: float = 0.7,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Smallest mean paired difference detectable at the given power.

    ``sd_unit`` is the between-unit SD of the score; ``rho`` the correlation
    between the two arms across units (paired designs on the same folds are
    strongly correlated, which is why the paired MDD is much smaller than the
    independent-samples one).
    """
    if n_units < 2:
        return float("nan")
    z_a = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_b = stats.norm.ppf(power)
    sd_diff = sd_unit * math.sqrt(max(2.0 * (1.0 - rho), 0.0))
    return float((z_a + z_b) * sd_diff / math.sqrt(n_units))


def mdd_paired_proportions(
    p: float,
    n_units: int,
    *,
    rho: float = 0.7,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """MDD for an accuracy near ``p``, using ``sqrt(p(1-p))`` as the unit SD.

    This is the calculation behind the pre-registered figure "video-level MDD
    ~= 8 accuracy points at 90 video units, p ~ 0.15": with p = 0.15, n = 90 and
    rho = 0.7 the formula gives 0.082.  The ``sqrt(p(1-p))`` proxy for the
    between-video SD is deliberately conservative; supply the empirical
    between-video SD to :func:`mdd_from_sd` once folds have been run.
    """
    return mdd_from_sd(
        math.sqrt(max(p * (1 - p), 0.0)), n_units,
        rho=rho, alpha=alpha, power=power, two_sided=two_sided,
    )


def mdd_table(
    *,
    p_primary: float = 0.15,
    n_subjects: int = 80,
    n_videos: int = 90,
    rho_subject: float = 0.80,
    rho_video: float = 0.70,
    sd_subject: Optional[float] = None,
    sd_video: Optional[float] = None,
    n_queries_per_subject: int = 144,
    n_queries_per_video: int = 640,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """Minimal detectable difference at both inference targets, as a range.

    Two bounds are reported per target, because the true resolution depends on
    the between-unit variance that only the folds can measure:

    * ``mdd`` -- the **conservative** bound, taking the unit SD to be
      ``sqrt(p(1-p))``.  This treats each unit as a single Bernoulli draw and is
      the figure the analysis plan pre-registers for the video target
      (~8 points at 90 videos, p ~ 0.15), because genuine between-video variance
      really is of that order.
    * ``mdd_floor`` -- the **measurement-error-only** bound, taking the unit SD
      to be ``sqrt(p(1-p) / n_queries_per_unit)`` (zero between-unit variance).
      No design can resolve less than this; the truth lies between the two.

    Supplying ``sd_subject`` / ``sd_video`` from real folds replaces the
    conservative proxy with the empirical SD, at which point ``mdd`` is the
    honest number and the range collapses.
    """
    rows = []
    for (target, unit, n_units, rho, sd_emp, n_q, test, claims) in (
        (
            "fixed stimuli (generalises over subjects)", "subject", n_subjects,
            rho_subject, sd_subject, n_queries_per_subject,
            "subject-level paired Wilcoxon (same folds)",
            "model ranking, ablations, encoder comparison",
        ),
        (
            "stimulus-generalising (subjects x videos)", "base video", n_videos,
            rho_video, sd_video, n_queries_per_video,
            "crossed subject x video mixed model (or by-video paired test)",
            "'holds for touch videos in general'",
        ),
    ):
        sd_cons = math.sqrt(max(p_primary * (1 - p_primary), 0.0))
        sd_used = sd_emp if sd_emp is not None else sd_cons
        sd_floor = sd_cons / math.sqrt(max(n_q, 1))
        rows.append(
            {
                "inference_target": target,
                "unit": unit,
                "n_units": n_units,
                "sd_unit": sd_used,
                "sd_source": "empirical" if sd_emp is not None else "sqrt(p(1-p)) proxy",
                "rho": rho,
                "n_queries_per_unit": n_q,
                "mdd": mdd_from_sd(sd_used, n_units, rho=rho, alpha=alpha, power=power),
                "mdd_floor": mdd_from_sd(sd_floor, n_units, rho=rho, alpha=alpha,
                                         power=power),
                "test": test,
                "claims": claims,
            }
        )
    df = pd.DataFrame(rows)
    df["mdd_points"] = df["mdd"] * 100.0
    df["mdd_floor_points"] = df["mdd_floor"] * 100.0
    df["alpha"] = alpha
    df["power"] = power
    df["p_assumed"] = p_primary
    return df


def print_mdd_table(
    table: Optional[pd.DataFrame] = None, **kwargs: Any
) -> pd.DataFrame:
    """Print the resolution of the design at both inference targets.

    Call this before reading any ablation table.  A difference below the
    relevant MDD is *not resolvable* by this design and must be reported that
    way -- neither as evidence of equivalence nor as a win.
    """
    df = table if table is not None else mdd_table(**kwargs)
    print("=" * 78)
    print("MINIMAL DETECTABLE DIFFERENCE -- read before interpreting any ablation")
    print("=" * 78)
    for _, r in df.iterrows():
        print(f"\n  target : {r['inference_target']}")
        print(f"  unit   : {r['unit']} (n = {int(r['n_units'])})")
        print(f"  test   : {r['test']}")
        print(
            f"  MDD    : {r['mdd_floor_points']:.1f} - {r['mdd_points']:.1f} "
            f"accuracy points (measurement-error floor - conservative bound; "
            f"alpha={r['alpha']}, power={r['power']}, rho={r['rho']}, "
            f"sd={r['sd_unit']:.3f} [{r['sd_source']}])"
        )
        print(f"  covers : {r['claims']}")
    print(
        "\n  Consequence: THINGS-style 1-4 point loss differences sit below the "
        "\n  video-level resolution of this design. Report them as unresolved."
    )
    print("=" * 78)
    return df


def flag_below_mdd(
    observed_diff: float, mdd: float, *, label: str = ""
) -> Dict[str, Any]:
    """Classify an observed difference against the design's resolution."""
    d = abs(float(observed_diff))
    resolvable = bool(np.isfinite(mdd) and d >= mdd)
    if not np.isfinite(mdd):
        verdict = "MDD unavailable"
    elif resolvable:
        verdict = "resolvable at this inference target"
    else:
        verdict = (
            f"BELOW RESOLUTION ({d*100:.1f} < {mdd*100:.1f} points): report as "
            "unresolved, not as a win and not as equivalence"
        )
    return {
        "label": label,
        "observed_diff": float(observed_diff),
        "mdd": float(mdd),
        "resolvable": resolvable,
        "verdict": verdict,
    }


def recommend_test(claim: str) -> Dict[str, str]:
    """Map a claim type to the test, the unit, and the resolution to quote.

    ``claim`` in {"model_ranking", "ablation", "stimulus_generalisation",
    "time_course", "phenotype"}.
    """
    table = {
        "model_ranking": {
            "inference_target": "fixed stimuli",
            "test": "paired_wilcoxon over subjects (same folds), n=80",
            "unit": "subject",
            "resolution": "quote the subject-level MDD",
        },
        "ablation": {
            "inference_target": "fixed stimuli",
            "test": "paired_wilcoxon over subjects (same folds), n=80",
            "unit": "subject",
            "resolution": "quote the subject-level MDD; if the claim is that the "
            "component helps *for touch videos in general*, escalate to the "
            "crossed model and the video-level MDD",
        },
        "stimulus_generalisation": {
            "inference_target": "stimulus-generalising",
            "test": "crossed_mixed_model (fallback: by_video_paired_test)",
            "unit": "base video",
            "resolution": "video-level MDD (~8 points at 90 videos, p~0.15)",
        },
        "time_course": {
            "inference_target": "depends on the claim; declare it",
            "test": "cluster_permutation_1d (subjects) or "
            "cluster_permutation_from_null (base videos)",
            "unit": "subject or base video",
            "resolution": "cluster p-values only; no pointwise onset latencies",
        },
        "phenotype": {
            "inference_target": "between-subject",
            "test": "attenuation-corrected correlation with permutation null",
            "unit": "subject",
            "resolution": "n=80 has 80% power only at r>=0.31; typical published "
            "effects are r~0.1-0.25, so treat as a supporting endpoint",
        },
    }
    if claim not in table:
        raise KeyError(f"unknown claim type {claim!r}; known: {sorted(table)}")
    return table[claim]
