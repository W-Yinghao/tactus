"""Correlated Component Analysis and inter-subject correlation.

CorrCA (Dmochowski et al. 2012; Parra lab) finds the spatial filters that
maximize correlation *between subjects*.  With 80 subjects who all saw the same
360 conditions this dataset is close to an ideal case, and the analysis buys two
things:

1. **A spatial-filter baseline.**  A handful of components that generalize
   across subjects without any learning is the floor any deep "subject-invariant
   embedding" has to clear.
2. **The per-subject ISC magnitude**, which the phenotype analysis needs as an
   SNR nuisance regressor.  A subject whose EEG simply correlates better with
   everyone else's will look "better aligned" under any model, and a phenotype
   claim that does not partial that out is a claim about cap fit.

Four things about this implementation are not cosmetic; each of them was a real,
measured bug in an earlier version, and each is documented here because the
number changes by an order of magnitude depending on the switch.

*What "shared stimulus" means (``remove_grand_erp``, default ON).*  Laying 360
condition-averages end to end gives a timeline that is identical across subjects
in two different senses.  Every condition-average contains the same
stimulus-*onset* response -- a video started, any video -- and that
condition-invariant ERP is by far the largest between-subject correlation in the
data.  Measured at n=80, w0600, decim 2: with the grand ERP left in, the top
component reaches per-pair ISC 0.1522, but *permuting each subject's 360
condition-averages independently* -- which destroys all stimulus correspondence
and leaves everything else intact -- still gives 0.1391, i.e. 91% of it.  The
quantity was almost entirely "a video started at t=0", not "this video".
Refitting on the grand ERP alone gives 0.6221; refitting on the
condition-specific residual gives 0.0191.  ``remove_grand_erp`` subtracts each
subject's mean-over-conditions before the covariances, so the headline is the
stimulus-*specific* ISC.  ``--keep-grand-erp`` restores the old, condition-
invariant quantity; it is a legitimate measurement of the onset response, and it
is labelled as such (``isc_kind``) rather than sold as stimulus decoding.

*The null has to be the stimulus-identity null.*  White noise tests nothing
time-locked, and a per-subject circular time shift de-phases the onset ERP,
i.e. it nulls out exactly the component that dominates -- both of those flatter
the result.  The only null that isolates stimulus correspondence is the
condition permutation above, so it is computed by default (``--null-perm``) and
reported next to the eigenvalues.  Against it the margin is what it is:
1.09x with the grand ERP in, 3.5x with it removed.

*One loud subject can own the fit (``amplitude_normalize``, default ON).*  R_w is
a *sum* over subjects, so a subject with a blown channel supplies most of the
whitening matrix and the filters bend to suppress it.  Measured here: sub-17
(channel PO4, sd 273.7 vs 3.31 median, survives ICA + autoreject + IQR-based
robust scaling because an IQR is insensitive to spikes) supplied **90.9% of
trace(R_w)**, and dropping that one subject moved per-pair c1 by +34%.  Shrinkage
does not help: it shrinks toward ``trace(R_w)/C * I``, which the same subject
dominates.  Dividing each subject's matrix by its own RMS makes every subject
contribute exactly 1/K of the trace and drops the drop-sub-17 sensitivity from
+34% to +2%.  The QC log line prints the per-subject amplitude and worst
channel-sd ratio and warns on outliers -- fix the recording, do not rely on the
normalizer.

*A scale-invariant per-subject ISC.*  The classic per-subject formula
``w'R_b^k w / w'R_w^k w`` is a ratio of ``2*sum<t_k,t_l>`` to
``sum(|t_k|^2 + |t_l|^2)``, which peaks when subject k's amplitude matches the
group and falls off on both sides: scaling one subject's data by 0.25 / 1 / 10
moved it 0.1353 / 0.2591 / 0.0463 while the truth held at 0.2590.  A regressor
that responds to microvolt gain *is* a claim about cap fit, so the advertised
per-subject quantity (``isc_r_c*``) is the mean pairwise Pearson correlation of
the component time course, which is invariant by construction.  The old ratio is
kept as ``isc_ratio_c*`` and is not scale-invariant.  For the same reason
``component_isc_per_pair`` (eigenvalue / (K-1)) is a *lower bound* on the mean
pairwise correlation, since 2ab <= a^2 + b^2: 0.1522 vs 0.1809 on the same data.
``component_isc_mean_pairwise_r`` is the unbiased-in-scale headline.

*Cost.*  The pooled between-subject covariance never requires the K x K matrix
of cross-covariances.  With ``S = sum_k X_k``,

    R_b = sum_{k != l} X_k' X_l = S'S - sum_k X_k' X_k

so one pass over subjects suffices, and the leave-one-subject-out quantities
follow from ``S - X_k`` by the same identity.  80 subjects cost 80 GEMMs, not
6400.

*Bias.*  A subject's ISC computed with filters that were themselves fitted using
that subject is inflated, and inflated *more* for subjects with more idiosyn-
cratic data.  ``--loso-filters`` (default on) refits the filters without each
subject before scoring it, which is the only version safe to correlate with a
questionnaire.  ``per_condition_isc_mean`` has no such protection and is
explicitly labelled in-sample.

    python -m tactus.baselines.corrca --n-components 5 --split-half
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import linalg as sla

from tactus.common import (
    EpochStore,
    atomic_write_json,
    condition_averages,
    load_channels,
    load_trials,
    results_dir,
    split_half_reliability,
    window_times_ms,
)

log = logging.getLogger(__name__)

__all__ = [
    "corrca_filters",
    "subject_isc",
    "component_timecourses",
    "pairwise_isc",
    "run_corrca",
    "main",
]


# --------------------------------------------------------------------------- #
# core algebra
# --------------------------------------------------------------------------- #


def _shrink(r: np.ndarray, gamma: float) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage toward a scaled identity.

    Note what this cannot do: the target ``trace(r)/C * I`` is itself set by
    whichever subject dominates ``trace(r)``, so shrinkage is no defence against
    one loud subject.  That is what ``amplitude_normalize`` is for.
    """
    c = r.shape[0]
    return (1.0 - gamma) * r + gamma * (np.trace(r) / c) * np.eye(c)


def corrca_filters(
    r_w: np.ndarray, r_b: np.ndarray, n_components: int = 5, gamma: float = 0.05
) -> tuple:
    """Solve the CorrCA generalized eigenproblem.

    Returns ``(W, isc, A)``: filters ``(C, n_components)``, their ISC values
    (the generalized eigenvalues -- ``w'R_b w / w'R_w w``), and the forward
    models ``A = R_w W (W'R_w W)^-1`` that are what you may actually plot as a
    scalp map.  Plotting the filters themselves is the classic error: a filter
    weight can be large purely to *suppress* a noise source.
    """
    r_w_s = _shrink(np.asarray(r_w, dtype=np.float64), gamma)
    r_b_s = np.asarray(r_b, dtype=np.float64)
    r_b_s = 0.5 * (r_b_s + r_b_s.T)
    vals, vecs = sla.eigh(r_b_s, r_w_s)
    order = np.argsort(vals)[::-1][:n_components]
    w = vecs[:, order]
    w = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
    isc = vals[order]
    m = w.T @ r_w_s @ w
    a = r_w_s @ w @ np.linalg.pinv(m)
    return w.astype(np.float64), isc.astype(np.float64), a.astype(np.float64)


def subject_isc(
    w: np.ndarray, r_kk: np.ndarray, r_sum: np.ndarray, r_w: np.ndarray, n_subjects: int
) -> np.ndarray:
    """Per-subject ISC ratio for each component (Cohen & Parra 2016).

    ``ISC_k = (w' R_b^k w) / (w' R_w^k w)`` with
    ``R_b^k = X_k'(S - X_k) + (S - X_k)'X_k`` and
    ``R_w^k = (K-1) R_kk + (R_w - R_kk)``.

    **Not scale-invariant, and therefore not the SNR nuisance regressor.**  The
    numerator is ``2 * sum_l <t_k, t_l>`` and the denominator
    ``sum_l (|t_k|^2 + |t_l|^2)``, so the value depends on subject k's absolute
    amplitude: it is maximal when k matches the group amplitude and decays on
    both sides (measured: x0.25 -> 0.1353, x1 -> 0.2591, x10 -> 0.0463, while
    the mean pairwise correlation stays at 0.2590).  ``2ab <= a^2 + b^2`` also
    makes it a *lower bound* on the mean pairwise correlation.  Use
    :func:`pairwise_isc` for the quantity you can regress a questionnaire on;
    this one is kept for continuity with the CorrCA literature.
    """
    k = int(n_subjects)
    cross = r_sum - r_kk  # X_k' (S - X_k)
    r_b_k = cross + cross.T
    r_w_k = (k - 1) * r_kk + (r_w - r_kk)
    num = np.einsum("ci,cd,di->i", w, r_b_k, w, optimize=True)
    den = np.einsum("ci,cd,di->i", w, r_w_k, w, optimize=True)
    return num / np.maximum(den, 1e-20)


def component_timecourses(w: np.ndarray, x_stack: np.ndarray) -> np.ndarray:
    """Project every subject onto ``w`` and z-normalize each time course.

    ``x_stack`` is ``(K, n, C)``, ``w`` is ``(C, m)``; returns ``(K, n, m)`` with
    each subject/component time course centred and scaled to unit norm.  Any
    per-subject gain cancels here, which is the whole point: everything computed
    downstream from this array is scale-invariant.
    """
    k, n, c = x_stack.shape
    y = (x_stack.reshape(k * n, c) @ np.asarray(w, np.float64)).reshape(k, n, -1)
    y = y - y.mean(axis=1, keepdims=True)
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def pairwise_isc(y: np.ndarray) -> np.ndarray:
    """Mean pairwise Pearson r of the component time course, per subject.

    ``y`` is the ``(K, n, m)`` output of :func:`component_timecourses`.  Returns
    ``(K, m)``: entry ``(k, j)`` is ``mean_{l != k} corr(y_kj, y_lj)``.  Its mean
    over subjects is the mean over all ordered pairs, i.e. the honest group ISC
    that ``component_isc_per_pair`` only lower-bounds.

    Computed from ``sum_l y_l`` rather than the K x K correlation matrix, so it
    costs one pass, not 6400.
    """
    k = y.shape[0]
    if k < 2:
        return np.zeros((k, y.shape[2]))
    s = y.sum(axis=0)  # (n, m)
    return (np.einsum("knm,nm->km", y, s, optimize=True) - 1.0) / (k - 1)


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #


def _subject_matrix(
    avg: np.ndarray,
    conditions: np.ndarray,
    center: bool = True,
    *,
    remove_grand_erp: bool = True,
    amplitude_normalize: bool = True,
) -> tuple:
    """``(n_conditions, 64, T)`` condition averages -> ``((n_cond * T, 64), qc)``.

    Concatenating the condition-averaged epochs is the event-related analogue of
    CorrCA's continuous time-locked data: the "shared stimulus timeline" is the
    evoked responses laid end to end, identical across subjects by construction.

    Two corrections happen here, both documented at module level:

    ``remove_grand_erp``
        subtract this subject's mean over the selected conditions, timepoint by
        timepoint.  Without it the concatenated timeline is dominated by the
        condition-*invariant* stimulus-onset response, which is shared across
        subjects for reasons that have nothing to do with which video was shown
        (measured: 91% of the top component survives a condition permutation).
        Centring over the whole concatenated timeline -- what ``center`` does --
        removes only the DC offset and leaves that ERP completely intact.

    ``amplitude_normalize``
        divide by the RMS of the matrix that actually enters the covariance, so
        every subject contributes exactly ``1/K`` of ``trace(R_w)``.  Without it
        sub-17 alone supplied 90.9%.

    ``qc`` carries the diagnostics for the log line: raw RMS amplitude, the
    worst channel sd relative to that subject's median channel, and the largest
    absolute sample in units of the median channel sd.
    """
    x = np.asarray(avg[conditions], dtype=np.float64)  # (n_cond, C, T)
    if remove_grand_erp:
        x = x - x.mean(axis=0, keepdims=True)
    x = np.transpose(x, (0, 2, 1)).reshape(-1, x.shape[1])
    if center:
        x -= x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    med = float(np.median(sd))
    amp = float(np.sqrt(np.mean(x**2)))
    qc = {
        "amplitude_rms": amp,
        "channel_sd_max": float(sd.max()),
        "channel_sd_median": med,
        "channel_sd_ratio": float(sd.max() / max(med, 1e-12)),
        "max_abs_over_median_sd": float(np.abs(x).max() / max(med, 1e-12)),
        "worst_channel": int(np.argmax(sd)),
    }
    if amplitude_normalize:
        x = x / max(amp, 1e-12)
    return x, qc


def _gather_condition_averages(
    subjects: Sequence[int],
    trials: pd.DataFrame,
    store: EpochStore,
    *,
    split: str,
    decim: int,
    cache_dir: Path,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    counts: Dict[int, np.ndarray] = {}
    for sid in subjects:
        avg, cnt = condition_averages(
            trials, store, subject_id=int(sid), split=split, decim=decim,
            cache_dir=cache_dir,
        )
        out[int(sid)] = avg
        counts[int(sid)] = cnt
        # EpochStore caches every memmap it opens; over 80 subjects the resident
        # pages (88 MB/subject at w0600, 132 MB at wm100_800) dominate RSS even
        # though we only ever need one subject at a time.  Numerically a no-op.
        store.close()
    if not counts:
        raise RuntimeError("no subjects were given to CorrCA")
    common = np.ones(next(iter(counts.values())).shape[0], dtype=bool)
    for cnt in counts.values():
        common &= cnt > 0
    log.info("%d/%d conditions present for all %d subjects",
             int(common.sum()), common.size, len(subjects))
    out["__conditions__"] = np.flatnonzero(common)  # type: ignore[assignment]
    return out


def _fit_from_stack(
    x_stack: np.ndarray, *, n_components: int, gamma: float
) -> tuple:
    """``(K, n, C)`` -> ``(W, isc, A, r_kk, s_sum, r_w)`` via the S'S identity."""
    k = x_stack.shape[0]
    r_kk = np.stack([x_stack[i].T @ x_stack[i] for i in range(k)])
    s_sum = x_stack.sum(axis=0)
    r_w = r_kk.sum(axis=0)
    r_b = s_sum.T @ s_sum - r_w
    w, isc, a = corrca_filters(r_w, r_b, n_components=n_components, gamma=gamma)
    return w, isc, a, r_kk, s_sum, r_w


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run_corrca(
    subjects: Sequence[int],
    trials: pd.DataFrame,
    store: EpochStore,
    *,
    n_components: int = 5,
    gamma: float = 0.05,
    decim: int = 2,
    loso_filters: bool = True,
    per_condition: bool = False,
    split_half: bool = False,
    remove_grand_erp: bool = True,
    amplitude_normalize: bool = True,
    n_null: int = 10,
    null_seed: int = 0,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    """Fit CorrCA over ``subjects`` and score every subject.

    With the defaults the headline ``component_isc*`` is the *stimulus-specific*
    ISC (grand ERP removed) of an amplitude-equalized group, reported next to a
    condition-permutation null.  ``remove_grand_erp=False`` reproduces the older,
    condition-invariant number; the result dict records which one it is under
    ``isc_kind``.
    """
    t0 = time.time()
    cache_dir = cache_dir or (results_dir() / "cache" / "cond_avg")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if out_dir is not None:
        # created up-front: --per-condition writes its CSV before the final
        # save block, and would otherwise fail on a fresh output directory.
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    data = _gather_condition_averages(
        subjects, trials, store, split="all", decim=decim, cache_dir=cache_dir
    )
    conditions = np.asarray(data.pop("__conditions__"))
    if conditions.size < 10:
        raise RuntimeError(
            f"only {conditions.size} conditions survive in every subject; CorrCA "
            f"needs a shared timeline. Check the artifact-rejection drop rates."
        )

    subjects = [int(s) for s in subjects]
    k = len(subjects)
    c = int(next(iter(data.values())).shape[1])
    n_t = int(next(iter(data.values())).shape[2])
    isc_kind = "stimulus_specific" if remove_grand_erp else "condition_invariant_onset"

    mk = dict(remove_grand_erp=remove_grand_erp, amplitude_normalize=amplitude_normalize)
    qc_rows: List[Dict[str, Any]] = []
    x_stack = np.empty((k, conditions.size * n_t, c), dtype=np.float64)
    for i, sid in enumerate(subjects):
        x_stack[i], qc = _subject_matrix(data[sid], conditions, **mk)
        qc_rows.append({"subject_id": sid, **qc})
    qc_df = pd.DataFrame(qc_rows)

    # -- QC: say out loud which subjects are pathological ------------------- #
    amp = qc_df["amplitude_rms"].to_numpy(float)
    ratio = qc_df["channel_sd_ratio"].to_numpy(float)
    log.info(
        "amplitude QC: RMS median %.3g, range %.3g (sub-%02d) - %.3g (sub-%02d), "
        "max/median ratio %.1fx; channel-sd ratio median %.2f",
        float(np.median(amp)), float(amp.min()), int(qc_df["subject_id"][int(amp.argmin())]),
        float(amp.max()), int(qc_df["subject_id"][int(amp.argmax())]),
        float(amp.max() / max(amp.min(), 1e-12)), float(np.median(ratio)),
    )
    flagged = qc_df.loc[(ratio > 5.0) | (amp > 5.0 * np.median(amp))]
    for _, r in flagged.iterrows():
        log.warning(
            "QC sub-%02d: channel %d sd = %.1fx this subject's median channel "
            "(max|x| = %.0f median-sd), RMS = %.2fx the group median. This subject "
            "would supply %.1f%% of trace(R_w) unnormalized%s",
            int(r["subject_id"]), int(r["worst_channel"]), r["channel_sd_ratio"],
            r["max_abs_over_median_sd"], r["amplitude_rms"] / max(float(np.median(amp)), 1e-12),
            100.0 * r["amplitude_rms"] ** 2 / float(np.sum(amp**2)),
            "; amplitude_normalize is ON so it supplies 1/K instead"
            if amplitude_normalize else "; amplitude_normalize is OFF -- it does",
        )
    if not len(flagged):
        log.info("amplitude QC: no subject flagged")

    w, isc, a, r_kk, s_sum, r_w = _fit_from_stack(
        x_stack, n_components=n_components, gamma=gamma
    )
    trace_share = np.array([np.trace(r) for r in r_kk])
    trace_share = trace_share / max(float(trace_share.sum()), 1e-20)
    log.info("max single-subject share of trace(R_w): %.1f%% (sub-%02d); even split = %.1f%%",
             100 * trace_share.max(), subjects[int(trace_share.argmax())], 100.0 / k)

    y_full = component_timecourses(w, x_stack)
    mean_r_full = pairwise_isc(y_full).mean(axis=0)
    log.info("component ISC [%s] eigenvalues: %s", isc_kind,
             np.array2string(isc, precision=4, floatmode="fixed"))
    log.info("  per-pair (eig/(K-1), a LOWER bound): %s",
             np.array2string(isc / max(1, k - 1), precision=4, floatmode="fixed"))
    log.info("  mean pairwise r of the component time course (in-sample): %s",
             np.array2string(mean_r_full, precision=4, floatmode="fixed"))

    # -- the null that actually tests stimulus correspondence --------------- #
    null_isc = np.empty((0, n_components))
    null_mean_r = np.empty((0, n_components))
    if n_null > 0:
        buf = np.empty_like(x_stack)
        for rep in range(int(n_null)):
            rng = np.random.default_rng(int(null_seed) + rep)
            for i, sid in enumerate(subjects):
                # permute *which* condition lands in each slot of the timeline,
                # independently per subject.  Same data, same grand ERP, same
                # amplitude -- only the stimulus correspondence is destroyed.
                buf[i], _ = _subject_matrix(
                    data[sid], conditions[rng.permutation(conditions.size)], **mk
                )
            w_n, isc_n, _, _, _, _ = _fit_from_stack(
                buf, n_components=n_components, gamma=gamma
            )
            null_isc = np.vstack([null_isc, isc_n])
            null_mean_r = np.vstack(
                [null_mean_r, pairwise_isc(component_timecourses(w_n, buf)).mean(axis=0)]
            )
        del buf
        nm, ns = null_isc.mean(axis=0), null_isc.std(axis=0)
        log.info("condition-permutation null (n=%d, per-pair): %s  +- %s",
                 int(n_null), np.array2string(nm / max(1, k - 1), precision=4, floatmode="fixed"),
                 np.array2string(ns / max(1, k - 1), precision=4, floatmode="fixed"))
        log.info("  real / null ratio: %s",
                 np.array2string(isc / np.maximum(nm, 1e-12), precision=2, floatmode="fixed"))
        if isc[0] < 1.5 * nm[0]:
            log.warning(
                "component 1 is within 1.5x of the stimulus-identity null "
                "(%.4f vs %.4f per-pair): whatever this component is, it is not "
                "specific to which video was shown.",
                isc[0] / max(1, k - 1), nm[0] / max(1, k - 1),
            )

    # -- per-subject scoring ------------------------------------------------ #
    rows: List[Dict[str, Any]] = []
    for i, sid in enumerate(subjects):
        r_sum_k = x_stack[i].T @ s_sum
        if loso_filters:
            r_w_mk = r_w - r_kk[i]
            s_mk = s_sum - x_stack[i]
            r_b_mk = s_mk.T @ s_mk - r_w_mk
            w_k, _, _ = corrca_filters(r_w_mk, r_b_mk, n_components=n_components, gamma=gamma)
            y_k = component_timecourses(w_k, x_stack)
        else:
            w_k, y_k = w, y_full
        # advertised, scale-invariant: mean pairwise r against every other
        # subject, using filters that did not see this subject.
        r_vals = pairwise_isc(y_k)[i]
        ratio_vals = subject_isc(w_k, r_kk[i], r_sum_k, r_w, k)
        row: Dict[str, Any] = {
            "subject_id": sid,
            "isc_r_sum": float(np.sum(r_vals)),
            "isc_ratio_sum": float(np.sum(ratio_vals)),
            "amplitude_rms": float(qc_rows[i]["amplitude_rms"]),
            "channel_sd_ratio": float(qc_rows[i]["channel_sd_ratio"]),
        }
        row.update({f"isc_r_c{j+1}": float(v) for j, v in enumerate(r_vals)})
        row.update({f"isc_ratio_c{j+1}": float(v) for j, v in enumerate(ratio_vals)})
        rows.append(row)

    per_subject = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    loso_mean_r = per_subject[[f"isc_r_c{j+1}" for j in range(n_components)]].mean().to_numpy()
    lg = np.log10(np.maximum(per_subject["amplitude_rms"].to_numpy(float), 1e-30))
    if float(np.std(lg)) > 0:
        log.info(
            "corr(log10 amplitude, per-subject ISC c1): scale-invariant %+.3f vs "
            "ratio (old, NOT scale-invariant) %+.3f",
            float(np.corrcoef(lg, per_subject["isc_r_c1"])[0, 1]),
            float(np.corrcoef(lg, per_subject["isc_ratio_c1"])[0, 1]),
        )

    if split_half:
        rel: Dict[int, float] = {}
        rel_ss: Dict[int, float] = {}
        for sid in subjects:
            odd, _ = condition_averages(trials, store, subject_id=sid, split="odd",
                                        decim=decim, cache_dir=cache_dir)
            even, _ = condition_averages(trials, store, subject_id=sid, split="even",
                                         decim=decim, cache_dir=cache_dir)
            o, e = odd[conditions], even[conditions]
            rel[sid] = split_half_reliability(o, e)
            # same trap as the group ISC: a subject whose onset ERP is large looks
            # "reliable" even if nothing about it distinguishes the videos.
            rel_ss[sid] = split_half_reliability(
                o - np.nanmean(o, axis=0, keepdims=True),
                e - np.nanmean(e, axis=0, keepdims=True),
            )
            store.close()  # see _gather_condition_averages
        # map by subject_id, never positionally: ``per_subject`` was re-sorted
        # above and ``subjects`` need not arrive in ascending order.  Attaching
        # the list positionally silently permutes the SNR nuisance regressor.
        per_subject["split_half_reliability"] = per_subject["subject_id"].map(rel)
        per_subject["split_half_reliability_stimulus_specific"] = (
            per_subject["subject_id"].map(rel_ss)
        )
        for name, d in (("all", rel), ("stimulus-specific", rel_ss)):
            v = np.asarray([d[s] for s in subjects], dtype=np.float64)
            log.info("split-half reliability (%s): median %.3f (min %.3f, max %.3f)",
                     name, float(np.nanmedian(v)), float(np.nanmin(v)), float(np.nanmax(v)))

    result: Dict[str, Any] = {
        "window": str(store.window), "n_subjects": k, "subjects": list(subjects),
        "n_components": int(n_components),
        "n_conditions": int(conditions.size), "n_channels": c,
        "decim": decim, "gamma": gamma, "loso_filters": bool(loso_filters),
        "remove_grand_erp": bool(remove_grand_erp),
        "amplitude_normalize": bool(amplitude_normalize),
        "isc_kind": isc_kind,
        "component_isc": isc.tolist(),
        "component_isc_per_pair": (isc / max(1, k - 1)).tolist(),
        "component_isc_per_pair_note":
            "eigenvalue/(K-1); a LOWER bound on the mean pairwise correlation "
            "because 2ab <= a^2+b^2. Use component_isc_mean_pairwise_r.",
        "component_isc_mean_pairwise_r": mean_r_full.tolist(),
        "component_isc_mean_pairwise_r_note": "in-sample (full-data filter)",
        "component_isc_mean_pairwise_r_loso": loso_mean_r.tolist(),
        "max_subject_trace_share": float(trace_share.max()),
        "max_subject_trace_share_subject": int(subjects[int(trace_share.argmax())]),
        "qc_flagged_subjects": [int(s) for s in flagged["subject_id"]],
        "seconds": time.time() - t0,
    }
    if n_null > 0:
        nm = null_isc.mean(axis=0)
        result.update({
            "null_kind": "condition_permutation_per_subject",
            "null_n_perm": int(n_null),
            "null_component_isc": nm.tolist(),
            "null_component_isc_sd": null_isc.std(axis=0).tolist(),
            "null_component_isc_per_pair": (nm / max(1, k - 1)).tolist(),
            "null_component_isc_mean_pairwise_r": null_mean_r.mean(axis=0).tolist(),
            "isc_over_null": (isc / np.maximum(nm, 1e-12)).tolist(),
        })

    if per_condition:
        # NOTE: in-sample.  ``w`` was fitted on all K subjects and all conditions,
        # then applied to the same data, so the *level* of these numbers is
        # optimistic.  The *ranking* across conditions is still informative --
        # see per_condition_between_video_var_fraction.
        acc = np.zeros((n_components, conditions.size, n_t))
        for i in range(k):
            y = x_stack[i] @ w  # (n_cond*T, n_comp)
            y = y.reshape(conditions.size, n_t, n_components).transpose(2, 0, 1)
            y = y - y.mean(axis=2, keepdims=True)
            y = y / np.maximum(np.linalg.norm(y, axis=2, keepdims=True), 1e-12)
            acc += y
        # mean pairwise correlation from the sum of unit-norm time courses
        pc = ((acc ** 2).sum(axis=2) - k) / max(1, k * (k - 1))
        result["per_condition_isc_mean_in_sample"] = pc.mean(axis=1).tolist()
        result["per_condition_is_in_sample"] = True
        vid = (trials.drop_duplicates("condition_id")
                     .set_index("condition_id")["video_id"].reindex(conditions).to_numpy())
        ok = pd.notna(vid)
        if ok.sum() > 1 and pd.Series(vid[ok]).nunique() > 1:
            g = pd.Series(vid[ok]).to_numpy()
            frac = []
            for j in range(n_components):
                v = pc[j][ok]
                df = pd.DataFrame({"g": g, "v": v})
                mu = df.groupby("g")["v"].transform("mean").to_numpy()
                ss_b = float(np.sum((mu - v.mean()) ** 2))
                ss_t = float(np.sum((v - v.mean()) ** 2))
                frac.append(ss_b / max(ss_t, 1e-20))
            n_g = int(pd.Series(g).nunique())
            # under exchangeable noise E[SS_between/SS_total] = (n_groups-1)/(n-1)
            chance = (n_g - 1) / max(int(ok.sum()) - 1, 1)
            result["per_condition_between_video_var_fraction"] = frac
            result["per_condition_between_video_var_fraction_chance"] = float(chance)
            result["per_condition_n_videos"] = n_g
            log.info("per-condition ISC (IN-SAMPLE): between-video variance / total = "
                     "%s over %d videos (exchangeable-noise expectation %.3f)",
                     np.array2string(np.asarray(frac), precision=3, floatmode="fixed"),
                     n_g, chance)
        if out_dir is not None:
            pd.DataFrame(
                pc.T, columns=[f"c{i+1}" for i in range(n_components)]
            ).assign(condition_id=conditions).to_csv(out_dir / "per_condition_isc.csv",
                                                     index=False)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        per_subject.to_csv(out_dir / "per_subject_isc.csv", index=False)
        qc_df.to_csv(out_dir / "subject_qc.csv", index=False)
        try:
            channels = load_channels()
        except Exception:
            channels = [f"ch{i}" for i in range(c)]
        np.savez_compressed(
            out_dir / "components.npz", filters=w, forward=a, isc=isc,
            null_isc=null_isc, conditions=conditions, channels=np.asarray(channels),
            times_ms=window_times_ms(store.window)[::decim],
        )
        atomic_write_json(out_dir / "summary.json", result)
    result["per_subject"] = per_subject
    result["qc"] = qc_df
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tactus.baselines.corrca",
        description="Correlated Component Analysis / inter-subject correlation.",
    )
    p.add_argument("--window", default="w0600")
    p.add_argument("--subjects", default="all")
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.05, help="shrinkage on R_w")
    p.add_argument("--decim", type=int, default=2)
    p.add_argument("--loso-filters", dest="loso", action="store_true", default=True)
    p.add_argument("--no-loso-filters", dest="loso", action="store_false")
    p.add_argument("--keep-grand-erp", dest="remove_erp", action="store_false", default=True,
                   help="do NOT remove the condition-invariant ERP; measures the "
                        "stimulus-ONSET response shared by all conditions, which is "
                        "~8x larger and ~1.09x the stimulus-identity null")
    p.add_argument("--no-amplitude-normalize", dest="amp_norm", action="store_false",
                   default=True,
                   help="let each subject contribute in proportion to its own variance "
                        "(one blown channel then owns trace(R_w))")
    p.add_argument("--null-perm", type=int, default=10,
                   help="condition-permutation null repetitions (0 disables)")
    p.add_argument("--null-seed", type=int, default=0)
    p.add_argument("--per-condition", action="store_true",
                   help="also compute IN-SAMPLE ISC per condition (which stimuli "
                        "synchronize brains)")
    p.add_argument("--split-half", action="store_true",
                   help="per-subject odd/even reliability (the SNR nuisance regressor)")
    p.add_argument("--out", default=None)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from tactus.baselines.linear_mvpa import parse_subjects

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )
    out_dir = Path(args.out) if args.out else results_dir() / "baselines" / "corrca" / args.window
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = load_trials()
    store = EpochStore(args.window)
    available = sorted(set(trials["subject_id"].astype(int)) & set(store.available_subjects()))
    if not available:
        raise SystemExit(
            f"no subject has both trial-table rows and an epoch file for window "
            f"{args.window!r}; run the preprocessing stage first."
        )
    subjects = parse_subjects(args.subjects, available)
    if len(subjects) < 3:
        raise SystemExit(
            f"CorrCA needs at least 3 subjects (got {len(subjects)}): with 2 the "
            f"leave-one-subject-out filters are fitted on a single subject and the "
            f"between-subject covariance is rank-deficient."
        )
    log.info("CorrCA over %d subjects (window=%s, decim=%d, grand_erp_removed=%s, "
             "amplitude_normalized=%s)", len(subjects), args.window, args.decim,
             args.remove_erp, args.amp_norm)

    res = run_corrca(
        subjects, trials, store, n_components=args.n_components, gamma=args.gamma,
        decim=args.decim, loso_filters=args.loso, per_condition=args.per_condition,
        split_half=args.split_half, remove_grand_erp=args.remove_erp,
        amplitude_normalize=args.amp_norm, n_null=args.null_perm,
        null_seed=args.null_seed, out_dir=out_dir,
    )
    ps = res["per_subject"]
    log.info("per-subject ISC [%s, scale-invariant mean pairwise r, sum of %d components]: "
             "mean %.4f, sd %.4f, range %.4f-%.4f", res["isc_kind"], args.n_components,
             ps["isc_r_sum"].mean(), ps["isc_r_sum"].std(),
             ps["isc_r_sum"].min(), ps["isc_r_sum"].max())
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
