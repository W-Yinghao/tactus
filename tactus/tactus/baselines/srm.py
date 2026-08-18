"""Shared Response Model on trial-averaged condition responses.

The classical competitor to deep cross-subject alignment, and the one this
dataset is almost embarrassingly well suited to: SRM wants many subjects
observing an identical, densely repeated stimulus timeline, and ds005662 gives
80 subjects x 360 identical conditions x 8 repeats.

Deterministic SRM (Chen et al., NeurIPS 2015) factorizes each subject's
condition-response matrix as ``X_k ~ W_k S`` with ``W_k`` orthonormal and ``S``
shared, alternating

    S     = (1/K) sum_k W_k' X_k
    W_k   = U V'   from the SVD of  X_k S'      (orthogonal Procrustes)

**Enrollment is the interesting part.**  A new subject joins by solving only the
Procrustes step -- one SVD, no re-fit of ``S``, no gradient descent -- using its
responses to the *training* conditions.  That gives the honest classical answer
to "how many calibration trials does a new subject need?", which is the same
question the deep model's LOSO+k curve asks.  ``--enroll-fracs`` sweeps it.

**Per-feature standardisation is not optional here.**  The Frobenius objective
``sum_k ||X_k - W_k S||^2`` weights every feature by its raw variance, and on
this cohort the raw variance is pathological: measured over all 80 subjects at
``w0600 --decim 2``, sub-17 alone carries **91.2%** of the total energy of the
80 condition-average matrices, and **99.0%** of sub-17's own energy sits in the
single blown channel PO4 (channel sd 273.7 against a 3.31 median; it survived
ICA + autoreject into contract B, and the per-channel *robust* scaler in
``derived/scaling/`` normalises by IQR, so the spike survives that too).  The
trainer dataloader is protected by ``data.scaling.clamp``; this module reads the
memmaps through :func:`tactus.common.condition_averages` and is **not**.  An
un-standardised SRM therefore spends its shared space reconstructing one dead
electrode, and scores at or below chance.  ``--standardize zscore`` (the
default) divides each feature by its training-column sd and drops the top-3
subjects' share of the objective from 0.942 to 0.038 (uniform = 0.037).

**Every run also reports a no-SRM control.**  ``control_*`` columns run the
identical protocol -- same fold, same query subject, same gallery subjects, same
fold-safe centring and scaling, same scorer -- with the ``W``/``S`` projection
deleted: the query is the subject's own centred channel x time pattern and the
gallery is the mean of the training subjects' centred patterns.  It is the
honest floor for "did the alignment step do anything", it costs one extra cosine
per subject, and it cannot be switched off, because an SRM number published
without it is unfalsifiable.  As of the 80-subject measurement in this file's
history SRM **loses** to that control (see ``srm_vs_control`` in
``summary.json``); that is a finding, not a bug, and it must be reported as a
pair.

Two evaluations, deliberately kept apart:

``match``
    EEG-to-EEG condition matching: does the new subject's shared-space response
    to a held-out condition retrieve that condition among the training
    population's responses?  This is SRM's native benchmark.  Note it is *not*
    zero-shot in the stimulus sense -- the gallery is built from other subjects'
    responses to those same held-out videos.  The query subject is **always
    excluded** from the gallery average (``within_subject`` and ``loso`` both put
    the query subject inside the training population); leaving it in turns the
    benchmark into self-retrieval, which at K=4 subjects reads 0.28 top-1 against
    a 1/72 chance and decays as 1/K as subjects are added.
``video`` (``--to-video``)
    Shared space -> frozen video embedding by ridge, then cosine retrieval.
    This one *is* commensurable with the deep model and with ``linear_align``.
    Reported against two galleries: ``video`` (the held-out *conditions*, i.e.
    orientation counts as a distinct item) and ``video_base`` (the held-out
    *base videos*, the 18-way gallery the deep model's primary endpoint uses).

    Both cosines are taken **after subtracting the training-condition mean
    embedding** from prediction and gallery alike.  SigLIP2's 90 clips sit at a
    mean pairwise cosine of 0.88, so a ridge prediction is
    ``y_mean_train + small deviation``; scoring the raw prediction ranks the
    gallery identically for every query and returns exactly ``1/n_gallery``
    whatever the EEG says.  The mean is estimated on training conditions only,
    so no test information enters.

    python -m tactus.baselines.srm --k 20 --to-video --enroll-fracs 0.1,0.25,0.5,1.0

Caveat on commensurability: the unit here is an ~8-repeat condition average, not
a single trial, so these numbers are an upper-SNR variant of the deep model's
pseudo-trial endpoint rather than a like-for-like replacement.

Regime note: ``loso`` folds put all 90 videos on *both* sides of the split, so S,
every ``W`` (the enrollment ``W`` included) and the ridge would all be fit on the
conditions being retrieved.  Those folds are skipped unless
``--allow-stimulus-overlap`` is passed, and any row produced that way is stamped
``stimulus_disjoint=False``.  ``within_subject`` and ``double_disjoint`` are
video-disjoint and need no such flag.  When the guard empties a run the process
writes ``summary.json`` with ``status="skipped"`` and exits **3**, so a scheduler
records a skip rather than a green empty run.

Cache safety: the output directory carries a hash of the *entire resolved
config* -- subject list, seed, decim, n_iter, enroll fractions, model tag,
alphas, gallery sizes and the overlap flag included -- so a 6-subject probe and
an 80-subject run can never share a ``folds/`` directory.  ``config.json`` is
written next to the folds and re-checked on every start.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from tactus.common import (
    EpochStore,
    atomic_write_json,
    condition_averages,
    hash_obj,
    load_trials,
    load_video_emb,
    results_dir,
)
from tactus.baselines.linear_align import RidgeAlpha, retrieval_np

log = logging.getLogger(__name__)

__all__ = ["DetSRM", "SkipFold", "build_subject_matrices", "feature_scale",
           "run_fold", "main"]

#: Exit code for "the run produced nothing on purpose" (see ``--regime loso``
#: without ``--allow-stimulus-overlap``).  Distinct from 1 so a scheduler can
#: tell a scientific skip from a crash, and non-zero so it can tell it from
#: success.
EXIT_SKIPPED = 3

STANDARDIZE_CHOICES = ("none", "zscore", "robust")


class SkipFold(RuntimeError):
    """A fold that carries no usable train/test cell under the current subset."""


#: Ridge penalties for the shared-space -> video-embedding map.  Deliberately the
#: same grid ``linear_align`` uses, so the two baselines differ in their features
#: and not in their penalty path.  In ``--feature-mode channels`` GCV pins to the
#: lower boundary and RidgeAlpha says so; that warning is expected here and the
#: fix is *not* to widen the grid downwards -- measured on 80 subjects, extending
#: it to 1e-3 moves GCV to the new boundary and costs ~1 point of 18-way top-1
#: (0.072 -> 0.065).  Override with --alphas if you want to explore it.
RIDGE_ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


# --------------------------------------------------------------------------- #
# deterministic SRM
# --------------------------------------------------------------------------- #


class DetSRM:
    """Deterministic Shared Response Model.

    ``X_k`` is ``(n_features, n_samples)`` per subject -- here features are
    channels x time and samples are conditions.  ``fit`` learns ``{W_k}`` and
    ``S``; :meth:`enroll` learns ``W`` for an unseen subject against a frozen
    ``S`` (the cheap-enrollment path this module exists to demonstrate).

    Implemented directly rather than via BrainIAK: the deterministic variant is
    twenty lines, and the dependency is not worth it for an EEG matrix that fits
    in cache.
    """

    def __init__(self, n_components: int = 20, n_iter: int = 20, seed: int = 0,
                 tol: float = 1e-6) -> None:
        self.k = int(n_components)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.tol = float(tol)
        self.w_: Dict[int, np.ndarray] = {}
        self.s_: np.ndarray | None = None
        self.loss_: List[float] = []

    @staticmethod
    def _procrustes(x: np.ndarray, s: np.ndarray) -> np.ndarray:
        """``argmin_W ||X - W S||_F`` subject to ``W'W = I``."""
        u, _, vt = np.linalg.svd(x @ s.T, full_matrices=False)
        return u @ vt

    def fit(self, data: Dict[int, np.ndarray]) -> "DetSRM":
        keys = sorted(data)
        if not keys:
            raise ValueError("DetSRM.fit got no subjects")
        f, n = data[keys[0]].shape
        if self.k > min(f, n):
            raise ValueError(
                f"n_components={self.k} exceeds min(n_features={f}, n_samples={n}); "
                f"lower --k, raise the number of enrollment conditions, or switch "
                f"--feature-mode (channels caps n_features at 64)."
            )
        rng = np.random.default_rng(self.seed)
        for key in keys:  # random orthonormal init (Chen et al.'s default)
            q, _ = np.linalg.qr(rng.standard_normal((f, self.k)))
            self.w_[key] = q
        prev = np.inf
        for it in range(self.n_iter):
            s = np.zeros((self.k, n))
            for key in keys:
                s += self.w_[key].T @ data[key]
            s /= len(keys)
            loss = 0.0
            for key in keys:
                self.w_[key] = self._procrustes(data[key], s)
                loss += float(np.sum((data[key] - self.w_[key] @ s) ** 2))
            self.loss_.append(loss)
            self.s_ = s
            # `it > 0` is load-bearing: on the first pass prev is inf, and both
            # sides of the comparison evaluate to inf, so the test would be
            # vacuously true and stop after one iteration with a random W.
            if it > 0 and abs(prev - loss) <= self.tol * max(1.0, abs(prev)):
                log.debug("SRM converged after %d iterations", it + 1)
                break
            prev = loss
        # The loop stores the S that *produced* the final W, which leaves S half
        # an iteration stale.  Everything downstream (enroll, the ridge, the
        # gallery) mixes S with W, so close the loop before returning.
        s = np.zeros((self.k, n))
        for key in keys:
            s += self.w_[key].T @ data[key]
        self.s_ = s / len(keys)
        return self

    def enroll(self, x: np.ndarray, s: np.ndarray | None = None) -> np.ndarray:
        """Fit ``W`` for a new subject against the frozen shared response."""
        s = self.s_ if s is None else s
        if s is None:
            raise RuntimeError("DetSRM.enroll called before fit")
        if x.shape[1] != s.shape[1]:
            raise ValueError(
                f"enrollment data has {x.shape[1]} samples but the shared response "
                f"has {s.shape[1]}; they must be the same conditions in the same order."
            )
        return self._procrustes(x, s)

    def transform(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Project a subject's data into the shared space: ``W' X``."""
        return w.T @ x


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #


def build_subject_matrices(
    subjects: Sequence[int],
    trials: pd.DataFrame,
    store: EpochStore,
    *,
    decim: int = 2,
    feature_mode: str = "channels_time",
    cache_dir: Path | None = None,
) -> tuple:
    """Per-subject sample matrices, the common conditions, and the column stride.

    Features are channels x time (``channels_time``, the default -- an evoked
    response is a spatiotemporal pattern and SRM should be free to rotate in
    both) or channels only (``channels``, with time folded into samples, the
    formulation closer to fMRI SRM).

    Returns ``(matrices, conditions, cols_per_condition)``.  The third element is
    the number of *columns* one condition occupies: 1 for ``channels_time`` and
    ``T`` for ``channels``, where the columns of condition ``j`` are
    ``j * T ... j * T + T - 1``.  Callers must route every condition index
    through :func:`condition_columns` -- indexing a ``channels``-mode matrix with
    bare condition ids selects time points, which silently mixes train and test
    conditions inside a single column block.
    """
    cache_dir = cache_dir or (results_dir() / "cache" / "cond_avg")
    cache_dir.mkdir(parents=True, exist_ok=True)
    avgs: Dict[int, np.ndarray] = {}
    common: np.ndarray | None = None
    for sid in subjects:
        avg, cnt = condition_averages(trials, store, subject_id=int(sid), split="all",
                                      decim=decim, cache_dir=cache_dir)
        avgs[int(sid)] = avg
        common = (cnt > 0) if common is None else (common & (cnt > 0))
        # EpochStore caches every memmap it opens, and on a cold condition-average
        # cache this loop touches all 80 of them (~88 MB/subject at w0600, ~7 GB
        # resident) even though only one subject is ever needed at a time.  Same
        # fix, same reason, as corrca._gather_condition_averages.  Numerically a
        # no-op: the next call re-opens the file it needs.
        store.close()
    assert common is not None
    conditions = np.flatnonzero(common)
    log.info("%d conditions common to all %d subjects", conditions.size, len(subjects))

    out: Dict[int, np.ndarray] = {}
    cols_per_condition = 1
    # `pop` rather than `.items()`: the float32 averages and the float64 sample
    # matrices are otherwise both resident (~1.3 GB over 80 subjects at w0600).
    for sid in sorted(avgs):
        avg = avgs.pop(sid)
        x = avg[conditions]  # (n_cond, C, T)
        if feature_mode == "channels_time":
            m = x.reshape(x.shape[0], -1).T  # (C*T, n_cond)
        elif feature_mode == "channels":
            m = np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)  # (C, n_cond*T)
            cols_per_condition = int(x.shape[2])
        else:
            raise ValueError(f"feature_mode must be channels_time|channels, got {feature_mode!r}")
        # A grand mean over *all* conditions would let held-out conditions shift
        # the origin; run_fold re-centers on the training columns of the fold.
        m = m - m.mean(axis=1, keepdims=True)
        out[int(sid)] = np.ascontiguousarray(m, dtype=np.float64)
    return out, conditions, cols_per_condition


# --------------------------------------------------------------------------- #
# fold evaluation
# --------------------------------------------------------------------------- #


def feature_scale(x_train: np.ndarray, mode: str = "zscore",
                  floor_frac: float = 1e-3) -> np.ndarray:
    """Per-feature scale estimated on **training columns only**.

    ``x_train`` is ``(n_features, n_train_columns)`` and already centred on those
    columns; the return is ``(n_features, 1)`` and is meant to divide the *whole*
    matrix, test columns included.  Estimating it on the training columns is what
    keeps it fold-safe -- a scale fitted on all columns would let a held-out
    condition set the units of its own feature.

    Why this exists at all: SRM minimises ``sum_k ||X_k - W_k S||_F^2``, an
    objective with no scale invariance whatsoever, so a single high-variance
    feature buys the whole shared space.  On this cohort that is not a
    hypothetical -- sub-17's blown PO4 carries 91% of the 80-subject objective.

    ``zscore`` divides by the sd; ``robust`` divides by ``IQR/1.349`` (the
    Gaussian-consistent sd estimate), which is what to reach for if you suspect a
    handful of conditions rather than a whole channel is doing the damage.
    Features whose scale is below ``floor_frac`` of the median are floored rather
    than divided, so a flat/interpolated channel is left small instead of being
    amplified into the objective.
    """
    if mode == "none":
        return np.ones((x_train.shape[0], 1), dtype=np.float64)
    if mode == "zscore":
        s = x_train.std(axis=1, keepdims=True)
    elif mode == "robust":
        q1, q3 = np.percentile(x_train, [25.0, 75.0], axis=1, keepdims=True)
        s = (q3 - q1) / 1.349
    else:
        raise ValueError(f"standardize must be one of {STANDARDIZE_CHOICES}, got {mode!r}")
    pos = s[s > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    return np.maximum(s, max(floor_frac * med, np.finfo(np.float64).tiny))


def _conditions_of_videos(videos: Sequence[int]) -> np.ndarray:
    v = np.asarray(videos, dtype=np.int64)
    return np.sort(((v[:, None] - 1) * 4 + np.arange(4)[None, :]).ravel())


def condition_columns(cond_pos: np.ndarray, cols_per_condition: int) -> np.ndarray:
    """Expand condition positions into matrix column indices.

    Identity for ``channels_time``; for ``channels`` each condition owns a
    contiguous block of ``T`` columns.
    """
    p = np.asarray(cond_pos, dtype=np.int64)
    if cols_per_condition <= 1:
        return p
    return (p[:, None] * int(cols_per_condition)
            + np.arange(int(cols_per_condition))[None, :]).ravel()


def _by_condition(z: np.ndarray, cols_per_condition: int) -> np.ndarray:
    """``(k, n_cond * T)`` shared response -> ``(n_cond, k * T)`` retrieval rows."""
    if cols_per_condition <= 1:
        return z.T
    k = z.shape[0]
    t = int(cols_per_condition)
    return z.reshape(k, -1, t).transpose(1, 0, 2).reshape(-1, k * t)


def run_fold(
    fold: Any,
    matrices: Dict[int, np.ndarray],
    conditions: np.ndarray,
    *,
    cols_per_condition: int = 1,
    n_components: int = 20,
    n_iter: int = 20,
    enroll_fracs: Sequence[float] = (1.0,),
    to_video: bool = False,
    video: Any = None,
    gallery_sizes: Sequence[int] = (2, 10, 18, 90),
    alphas: Sequence[float] = RIDGE_ALPHAS,
    allow_stimulus_overlap: bool = False,
    standardize: str = "zscore",
    subject_energy_norm: bool = True,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Fit SRM on the training cells and evaluate every held-out subject.

    Every row carries both arms: ``match/``, ``video/``, ``video_base/`` from the
    SRM shared space, and ``control_match/``, ``control_video/``,
    ``control_video_base/`` from the same protocol with the ``W``/``S``
    projection removed.  The control has no enrollment step, so its columns are
    constant along ``enroll_frac`` -- that flat line is the bar the SRM curve has
    to clear.
    """
    train_videos = np.asarray(getattr(fold, "train_videos"), dtype=np.int64)
    test_videos = np.asarray(getattr(fold, "test_videos"), dtype=np.int64)
    train_subjects = [int(s) for s in np.asarray(getattr(fold, "train_subjects"))
                      if int(s) in matrices]
    test_subjects = [int(s) for s in np.asarray(getattr(fold, "test_subjects"))
                     if int(s) in matrices]
    if not test_subjects:  # within-subject regime: the same subjects are tested
        test_subjects = list(train_subjects)
        if str(getattr(fold, "regime", "")) not in ("", "within_subject"):
            log.warning(
                "fold %s is a %s fold but none of its held-out subjects survived the "
                "--subjects subset; falling back to testing the training subjects "
                "(seen_in_training=True). Cross-subject transfer is NOT being measured.",
                getattr(fold, "video_fold_id", "?"), getattr(fold, "regime", "?"),
            )
    if len(train_subjects) < 2:
        raise SkipFold(
            f"fold has {len(train_subjects)} training subject(s) inside the requested "
            f"--subjects subset; SRM needs at least 2 to define a shared response."
        )

    pos = {int(c): i for i, c in enumerate(conditions)}
    train_cols = np.array([pos[c] for c in _conditions_of_videos(train_videos) if c in pos],
                          dtype=np.int64)
    test_cols = np.array([pos[c] for c in _conditions_of_videos(test_videos) if c in pos],
                         dtype=np.int64)
    if train_cols.size < n_components or test_cols.size < 2:
        raise SkipFold(
            f"fold has {train_cols.size} train and {test_cols.size} test conditions; "
            f"need >= n_components train columns."
        )
    n_overlap = int(np.intersect1d(train_cols, test_cols).size)
    if n_overlap and not allow_stimulus_overlap:
        # `loso` folds carry all 90 videos on both sides.  S, every W (including
        # the new subject's enrollment W) and the ridge would then all be fit on
        # the very conditions being retrieved: the numbers are in-sample and are
        # not comparable to any zero-shot endpoint in this repo.
        raise SkipFold(
            f"{n_overlap}/{test_cols.size} test conditions are also training conditions "
            f"(regime={getattr(fold, 'regime', '?')}).  SRM would be scored on the stimuli "
            f"it was fit on.  Use --regime within_subject / double_disjoint, or pass "
            f"--allow-stimulus-overlap to score the in-sample number deliberately."
        )
    if n_overlap:
        log.warning(
            "fold %s: %d/%d test conditions overlap the training conditions -- "
            "match/ and video/ metrics for this fold are IN-SAMPLE (stimulus_disjoint=False)",
            getattr(fold, "video_fold_id", "?"), n_overlap, test_cols.size,
        )
    trc = condition_columns(train_cols, cols_per_condition)
    tec = condition_columns(test_cols, cols_per_condition)

    # Fold-safe re-centering and re-scaling: both the origin and the units of the
    # feature space are estimated on training columns only, never on the held-out
    # conditions.  `scale` is what stops one blown electrode from owning the
    # Frobenius objective -- see feature_scale and the module docstring.
    subjects_here = sorted(set(train_subjects) | set(test_subjects))
    offset = {s: matrices[s][:, trc].mean(axis=1, keepdims=True) for s in subjects_here}
    scale = {s: feature_scale(matrices[s][:, trc] - offset[s], standardize)
             for s in subjects_here}
    # One materialisation of the fold-normalised data, shared by both arms, so the
    # SRM and the control can never differ by anything except the projection.
    norm = {s: (matrices[s] - offset[s]) / scale[s] for s in subjects_here}

    # Per-SUBJECT energy normalisation (DECISIONS D11).  feature_scale is a
    # per-FEATURE operation, and that turns out not to be enough: with
    # `--standardize robust` on the interpolated data one subject still carried
    # 19.7-28.3% of the objective across the five within_subject folds against a
    # uniform share of 1/80, because a robust (IQR-based) scale under-reads a
    # heavy-tailed subject and leaves its squared energy large after division.
    # Dividing each subject's training block by its own Frobenius norm makes the
    # share uniform by construction for *any* feature scaling, which is what the
    # decision actually asked for.  Estimated on training columns only, like
    # every other constant here.
    #
    # It buys a better fit and no retrieval: objective drop 0.122 -> 0.158, but
    # video 18-way top-1 0.0760 -> 0.0741 (chance 0.0556).  A correctness fix,
    # not a performance one -- do not quote it as the latter.
    if subject_energy_norm:
        fro = {s: float(np.linalg.norm(norm[s][:, trc])) for s in subjects_here}
        med_fro = float(np.median([v for v in fro.values() if v > 0]) or 1.0)
        norm = {s: norm[s] / max(fro[s], 1e-12 * med_fro) for s in subjects_here}

    # Energy audit -- how much of `sum_k ||X_k - W_k S||^2` any single subject can
    # buy.  Uniform by construction once subject_energy_norm is on, which is the
    # point of running it after the block above rather than before.  Measured on
    # 80 subjects at w0600 --decim 2, per-feature scaling only: 93.2% for sub-17
    # under `none`, 65.4% under `robust`; after sub-17's PO4 was interpolated,
    # `robust` still gave 19.7-28.3% across the five within_subject folds, which
    # is what motivated the per-subject step.
    energy = {s: float(np.sum(norm[s][:, trc] ** 2)) for s in train_subjects}
    e_tot = sum(energy.values()) or 1.0
    worst_sid = max(energy, key=lambda s: energy[s])
    worst_share = energy[worst_sid] / e_tot
    if worst_share > 4.0 / max(1, len(train_subjects)):
        log.warning(
            "fold %s: sub-%02d carries %.1f%% of the SRM objective (%d training "
            "subjects, uniform would be %.1f%%) -- the shared space is being spent "
            "on one subject. --standardize is %r; note that per-feature scaling "
            "alone has already been shown not to fix this, so the lever is "
            "subject_energy_norm.",
            getattr(fold, "video_fold_id", "?"), worst_sid, 100 * worst_share,
            len(train_subjects), 100.0 / len(train_subjects), standardize,
        )

    srm = DetSRM(n_components=n_components, n_iter=n_iter, seed=seed)
    srm.fit({s: norm[s][:, trc] for s in train_subjects})
    assert srm.s_ is not None
    loss_drop = (1.0 - srm.loss_[-1] / srm.loss_[0]) if srm.loss_ else float("nan")

    # Shared responses of the *training population* to the held-out conditions.
    # Kept per subject so the query subject can be dropped from its own gallery.
    proj_test = {s: srm.w_[s].T @ norm[s][:, tec] for s in train_subjects}
    proj_sum = np.sum(list(proj_test.values()), axis=0)  # (k, n_test_cols)
    # ---- no-SRM control: identical protocol with W'/S deleted.  Query = the
    # subject's own fold-normalised pattern; gallery = mean of the training
    # subjects' fold-normalised patterns over the same held-out conditions.
    ctrl_test = {s: norm[s][:, tec] for s in train_subjects}
    ctrl_sum = np.sum(list(ctrl_test.values()), axis=0)  # (n_features, n_test_cols)

    ridge = ctrl_ridge = None
    y_center = ctrl_y_center = base_center = None
    base_ids = base_true = None
    if to_video and video is not None:
        y = video.cond_emb[conditions[train_cols]]
        ridge = RidgeAlpha(alphas).fit(_by_condition(srm.s_, cols_per_condition), y)
        y_center = np.asarray(ridge.y_mean_, dtype=np.float64)  # train-condition mean
        # The control's regressor is the training-population mean pattern on the
        # training conditions -- the same quantity S estimates, in the original
        # feature space.  RidgeAlpha solves this in the dual, so f >> n is cheap.
        ctrl_train = np.sum([norm[s][:, trc] for s in train_subjects], axis=0) \
            / len(train_subjects)
        ctrl_ridge = RidgeAlpha(alphas).fit(_by_condition(ctrl_train, cols_per_condition), y)
        ctrl_y_center = np.asarray(ctrl_ridge.y_mean_, dtype=np.float64)
        # 18-way base-video gallery: one item per held-out source video, the unit
        # the deep model's primary endpoint is scored on.
        base_ids = np.unique(conditions[test_cols] // 4)          # 0-based video ids
        vpos = {int(v): i for i, v in enumerate(base_ids)}
        base_true = np.array([vpos[int(c) // 4] for c in conditions[test_cols]], dtype=np.int64)
        train_base = np.unique(conditions[train_cols] // 4)
        base_center = video.base_emb[train_base].mean(axis=0, keepdims=True).astype(np.float64)

    # `video_fold_id` is -1 for loso and `subject_fold_id` is None outside the
    # cross-subject regimes; SeedSequence rejects negatives, so shift both up.
    fold_key = [int(seed),
                int(getattr(fold, "video_fold_id", -1)) + 1,
                (int(fold.subject_fold_id) if getattr(fold, "subject_fold_id", None)
                 is not None else -1) + 1]
    rows: List[Dict[str, Any]] = []
    for sid in test_subjects:
        seen = sid in train_subjects
        n_gal = len(train_subjects) - (1 if seen else 0)
        if n_gal < 1:
            raise SkipFold("no gallery subjects left after excluding the query subject")
        gal = _by_condition((proj_sum - proj_test[sid]) / n_gal if seen
                            else proj_sum / n_gal, cols_per_condition)

        # ---- control arm, computed once per subject (no enrollment step) ----
        ctrl_gal = _by_condition((ctrl_sum - ctrl_test[sid]) / n_gal if seen
                                 else ctrl_sum / n_gal, cols_per_condition)
        ctrl_q = _by_condition(norm[sid][:, tec], cols_per_condition)
        ctrl: Dict[str, Any] = {}
        mc = retrieval_np(ctrl_q, ctrl_gal, np.arange(test_cols.size),
                          gallery_sizes=gallery_sizes, seed=seed)
        ctrl.update({f"control_match/{k}": v for k, v in mc.items()
                     if not isinstance(v, dict)})
        if ctrl_ridge is not None:
            cdev = ctrl_ridge.predict(ctrl_q) - ctrl_y_center
            mcv = retrieval_np(
                cdev, video.cond_emb[conditions[test_cols]] - ctrl_y_center,
                np.arange(test_cols.size), gallery_sizes=gallery_sizes, seed=seed,
            )
            ctrl.update({f"control_video/{k}": v for k, v in mcv.items()
                         if not isinstance(v, dict)})
            mcb = retrieval_np(cdev, video.base_emb[base_ids] - base_center, base_true,
                               gallery_sizes=gallery_sizes, seed=seed)
            ctrl.update({f"control_video_base/{k}": v for k, v in mcb.items()
                         if not isinstance(v, dict)})
            ctrl["control_ridge_alpha"] = float(ctrl_ridge.alpha_)

        # One permutation per (fold, subject): the enrollment fractions are then
        # nested prefixes, so the curve is "add calibration conditions" and not
        # "resample which conditions you got".
        perm = np.random.default_rng(fold_key + [int(sid)]).permutation(train_cols.size)
        for frac in sorted(float(f) for f in enroll_fracs):
            n_enroll = max(n_components, int(round(frac * train_cols.size)))
            n_enroll = min(n_enroll, train_cols.size)
            sel = np.sort(perm[:n_enroll])
            sel_cols = condition_columns(sel, cols_per_condition)
            w = srm.enroll(norm[sid][:, trc[sel_cols]], srm.s_[:, sel_cols])
            z = w.T @ norm[sid][:, tec]                      # (k, n_test_cols)
            zc = _by_condition(z, cols_per_condition)        # (n_test_cond, k*T)

            row: Dict[str, Any] = {
                "subject_id": sid, "seen_in_training": bool(seen),
                "enroll_frac": float(frac), "n_enroll_conditions": int(n_enroll),
                "video_fold_id": int(getattr(fold, "video_fold_id", -1)),
                "subject_fold_id": (None if getattr(fold, "subject_fold_id", None) is None
                                    else int(fold.subject_fold_id)),
                "n_test_conditions": int(test_cols.size),
                "n_train_subjects": len(train_subjects),
                "n_gallery_subjects": int(n_gal),
                "gallery_excludes_self": True,
                "stimulus_disjoint": bool(n_overlap == 0),
                "regime": str(getattr(fold, "regime", "?")),
                "k": int(n_components), "standardize": str(standardize),
                "subject_energy_norm": bool(subject_energy_norm),
                "srm_objective_drop": float(loss_drop),
                "srm_n_iter_run": int(len(srm.loss_)),
                "worst_subject_objective_share": float(worst_share),
                "control_uses_enrollment": False,
            }
            m = retrieval_np(
                zc, gal, np.arange(test_cols.size),
                gallery_sizes=gallery_sizes, seed=seed,
            )
            row.update({f"match/{k}": v for k, v in m.items() if not isinstance(v, dict)})
            row.update(ctrl)
            if ridge is not None:
                row["ridge_alpha"] = float(ridge.alpha_)
                row["ridge_alpha_at_boundary"] = bool(getattr(ridge, "at_boundary_", False))
                dev = ridge.predict(zc) - y_center
                mv = retrieval_np(
                    dev, video.cond_emb[conditions[test_cols]] - y_center,
                    np.arange(test_cols.size), gallery_sizes=gallery_sizes, seed=seed,
                )
                row.update({f"video/{k}": v for k, v in mv.items() if not isinstance(v, dict)})
                mb = retrieval_np(
                    dev, video.base_emb[base_ids] - base_center, base_true,
                    gallery_sizes=gallery_sizes, seed=seed,
                )
                row["n_test_videos"] = int(base_ids.size)
                row.update({f"video_base/{k}": v for k, v in mb.items()
                            if not isinstance(v, dict)})
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tactus.baselines.srm",
        description="Shared Response Model with cheap new-subject enrollment.",
    )
    p.add_argument("--regime", default="double_disjoint",
                   choices=["within_subject", "loso", "double_disjoint"])
    p.add_argument("--window", default="w0600")
    p.add_argument("--subjects", default="all")
    p.add_argument("--k", type=int, default=20, help="number of shared components")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--decim", type=int, default=2)
    p.add_argument("--feature-mode", default="channels_time",
                   choices=["channels_time", "channels"])
    p.add_argument("--standardize", default="zscore", choices=list(STANDARDIZE_CHOICES),
                   help="per-feature scaling estimated on the fold's training "
                        "columns before SRM. 'none' reproduces the pre-2026-08 "
                        "behaviour in which sub-17's blown PO4 carries 91%% of the "
                        "objective and SRM scores at or below chance.")
    p.add_argument("--no-subject-energy-norm", dest="subject_energy_norm",
                   action="store_false",
                   help="skip the per-SUBJECT Frobenius normalisation. Per-feature "
                        "scaling alone is not sufficient: under 'robust' one "
                        "subject still carried 20-28%% of the objective across the "
                        "five within_subject folds (DECISIONS D11).")
    p.add_argument("--enroll-fracs", default="1.0",
                   help="comma-separated fractions of training conditions used to "
                        "enroll a new subject, e.g. 0.1,0.25,0.5,1.0")
    p.add_argument("--to-video", action="store_true",
                   help="also score shared-space -> video-embedding retrieval")
    p.add_argument("--model-tag", default="siglip2-base",
                   help="video embedding cache tag under data/derived/video_emb")
    p.add_argument("--n-video-folds", type=int, default=5)
    p.add_argument("--n-subject-folds", type=int, default=8)
    p.add_argument("--folds", type=lambda s: [int(x) for x in s.split(",") if x], default=None)
    p.add_argument("--gallery-sizes", default="2,10,18,90")
    p.add_argument("--alphas", default=",".join(f"{a:g}" for a in RIDGE_ALPHAS),
                   help="ridge penalty grid for the shared-space -> video map")
    p.add_argument("--allow-stimulus-overlap", action="store_true",
                   help="score folds whose test conditions were also training "
                        "conditions (--regime loso). The resulting match/ and video/ "
                        "numbers are IN-SAMPLE and are not zero-shot.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def resolved_config(args: argparse.Namespace, subjects: Sequence[int],
                    fracs: Sequence[float], sizes: Sequence[int],
                    alphas: Sequence[float]) -> Dict[str, Any]:
    """Every knob that can move a number, in one JSON-able dict.

    This is both the cache key (hashed into the output directory name) and the
    ``config`` block of ``summary.json``.  It exists because the old directory
    key was ``regime_window_k_featuremode`` only: a 6-subject probe and an
    80-subject run landed in the same ``folds/`` directory, the 80-subject run
    logged "cached" for the probe's folds, and ``summary.json`` still stamped
    ``regime=double_disjoint`` over a per_subject_folds.csv that contained
    ``n_train_subjects=6, seen_in_training=True`` rows.

    ``--folds``, ``--out``, ``--force`` and ``--log-level`` are deliberately
    excluded: they select *which* folds get computed and where, not *what* a
    fold computes, and the per-fold JSONs are keyed by fold index anyway.
    """
    return {
        "regime": args.regime,
        "window": args.window,
        "subjects": [int(s) for s in subjects],
        "n_subjects": len(subjects),
        "k": int(args.k),
        "n_iter": int(args.n_iter),
        "decim": int(args.decim),
        "feature_mode": args.feature_mode,
        "standardize": args.standardize,
        "subject_energy_norm": bool(args.subject_energy_norm),
        "enroll_fracs": [float(f) for f in fracs],
        "to_video": bool(args.to_video),
        "model_tag": args.model_tag if args.to_video else None,
        "n_video_folds": int(args.n_video_folds),
        "n_subject_folds": int(args.n_subject_folds),
        "gallery_sizes": [int(s) for s in sizes],
        "alphas": [float(a) for a in alphas],
        "allow_stimulus_overlap": bool(args.allow_stimulus_overlap),
        "seed": int(args.seed),
        "folds_from": "tactus.data.splits.make_folds",
    }


def main(argv: Sequence[str] | None = None) -> int:
    from tactus.baselines.linear_mvpa import parse_subjects
    from tactus.data.splits import make_folds

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )
    trials = load_trials()
    store = EpochStore(args.window)
    available = sorted(set(trials["subject_id"].astype(int)) & set(store.available_subjects()))
    subjects = parse_subjects(args.subjects, available)
    if len(subjects) < 2:
        log.error("SRM needs >= 2 subjects; --subjects %r resolved to %s",
                  args.subjects, subjects)
        return 2
    fracs = sorted({float(f) for f in args.enroll_fracs.split(",") if f.strip()})
    sizes = [int(s) for s in args.gallery_sizes.split(",")]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    cfg = resolved_config(args, subjects, fracs, sizes, alphas)
    cfg_hash = hash_obj(cfg, 10)

    out_dir = Path(args.out) if args.out else results_dir() / "baselines" / "srm"
    out_dir = out_dir / (f"{args.regime}_{args.window}_k{args.k}_{args.feature_mode}"
                         f"_{args.standardize}_s{len(subjects)}_{cfg_hash}")
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)
    log.info("out_dir %s", out_dir)

    # Belt and braces on top of the hashed directory name: if a config.json is
    # already here and disagrees, we have a hash collision (or a hand-edited
    # tree) and reusing the folds would silently mix configurations again.
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("config") != cfg:
            log.error("%s holds a DIFFERENT config under the same hash; refusing to "
                      "reuse its folds. Delete the directory or pass --out.", out_dir)
            return 2
    atomic_write_json(cfg_path, {"config_hash": cfg_hash, "config": cfg})

    video = load_video_emb(args.model_tag) if args.to_video else None

    t0 = time.time()
    matrices, conditions, cpc = build_subject_matrices(
        subjects, trials, store, decim=args.decim, feature_mode=args.feature_mode
    )
    log.info("built %d subject matrices %s (%d column(s) per condition) in %.1f s",
             len(matrices), matrices[subjects[0]].shape, cpc, time.time() - t0)

    # `subjects=` matters: without it the subject partition is built over the full
    # cohort, so on a subset run every held-out subject can fall outside the
    # subset and run_fold quietly falls back to testing the training subjects
    # (seen_in_training=True).  linear_align passes it; so do we.
    try:
        folds = make_folds(trials, args.regime, n_video_folds=args.n_video_folds,
                           n_subject_folds=args.n_subject_folds, seed=args.seed,
                           subjects=subjects)
    except ValueError as exc:
        # The common case is --n-subject-folds > len(--subjects).  Failing loudly
        # is the point: the alternative the old code had was to build the
        # partition over the full cohort and then quietly test the training
        # subjects.  Say what to change rather than raising a traceback.
        log.error("make_folds(%s, subjects=%d of them) refused the request: %s\n"
                  "Lower --n-subject-folds (currently %d) to at most the number of "
                  "subjects, or widen --subjects.",
                  args.regime, len(subjects), exc, args.n_subject_folds)
        atomic_write_json(out_dir / "summary.json", {
            "status": "skipped", "n_rows": 0, "n_folds": 0, "n_folds_skipped": 0,
            "skip_reason": f"make_folds refused the request: {exc}",
            "skip_reasons": [f"make_folds refused the request: {exc}"],
            "config_hash": cfg_hash, "config": cfg,
        })
        return EXIT_SKIPPED

    all_rows: List[Dict[str, Any]] = []
    n_skipped = 0
    skip_reasons: List[str] = []
    for i, fold in enumerate(folds):
        if args.folds and i not in args.folds:
            continue
        fpath = out_dir / "folds" / f"fold{i:03d}.json"
        if fpath.exists() and not args.force:
            with open(fpath, "r", encoding="utf-8") as fh:
                rows = json.load(fh)
            all_rows.extend(rows)
            log.info("fold %03d: cached", i)
            continue
        t1 = time.time()
        try:
            rows = run_fold(
                fold, matrices, conditions, cols_per_condition=cpc,
                n_components=args.k, n_iter=args.n_iter,
                enroll_fracs=fracs, to_video=args.to_video, video=video,
                gallery_sizes=sizes, alphas=alphas, seed=args.seed,
                allow_stimulus_overlap=args.allow_stimulus_overlap,
                standardize=args.standardize,
                subject_energy_norm=args.subject_energy_norm,
            )
        except SkipFold as exc:
            n_skipped += 1
            skip_reasons.append(f"fold{i:03d}: {exc}")
            log.warning("fold %03d: skipped -- %s", i, exc)
            continue
        for r in rows:
            r["fold_index"] = i
        atomic_write_json(fpath, rows)
        all_rows.extend(rows)
        df = pd.DataFrame(rows)
        msg = (f"fold {i:03d}: {len(rows)} subject-rows, match top1={df['match/top1'].mean():.4f} "
               f"(no-SRM control {df['control_match/top1'].mean():.4f})")
        if "video_base/top1" in df.columns:
            msg += (f", video_base 18-way top1={df['video_base/top1'].mean():.4f} "
                    f"(control {df['control_video_base/top1'].mean():.4f})")
        log.info("%s (%.1f s)", msg, time.time() - t1)

    if n_skipped:
        log.warning("%d fold(s) skipped; see warnings above", n_skipped)

    # ---- a run that produced nothing is a skip, not a success ---------------- #
    if not all_rows:
        reason = (skip_reasons[0] if skip_reasons else
                  "no fold index matched --folds, or make_folds returned nothing")
        atomic_write_json(out_dir / "summary.json", {
            "status": "skipped",
            "n_rows": 0, "n_folds": 0, "n_folds_skipped": n_skipped,
            "skip_reason": reason, "skip_reasons": skip_reasons,
            "config_hash": cfg_hash, "config": cfg,
        })
        log.error(
            "NO ROWS PRODUCED -- every requested fold was skipped (%d). Reason: %s\n"
            "Wrote summary.json with status=skipped and exiting %d so this is not "
            "recorded as a successful run.", n_skipped, reason, EXIT_SKIPPED,
        )
        return EXIT_SKIPPED

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_subject_folds.csv", index=False)
    # `enroll_frac` is itself a float column: leaving it in the aggregate list
    # makes reset_index try to re-insert a column that already exists.  The
    # int columns must stay in (n_enroll_conditions is the curve's x axis),
    # so only the identifiers are dropped.
    #
    # "b" is load-bearing.  `seen_in_training` and `stimulus_disjoint` are bool,
    # and the old `kind in "fi"` filter dropped them, so a curve computed over
    # rows in which the query subject was inside its own training population
    # looked exactly like a clean one.  Averaged as floats they read as
    # contamination fractions.
    skip = {"enroll_frac", "subject_id", "fold_index",
            "video_fold_id", "subject_fold_id"}
    num = [c for c in df.columns if df[c].dtype.kind in "fiub" and c not in skip]
    agg = df[["enroll_frac"] + num].copy()
    for c in num:
        if agg[c].dtype.kind == "b":
            agg[c] = agg[c].astype(np.float64)
    curve = agg.groupby("enroll_frac", observed=True)[num].mean().reset_index()
    curve.to_csv(out_dir / "enrollment_curve.csv", index=False)

    # ---- contamination audit: never let these hide in a bool column --------- #
    seen = df.get("seen_in_training", pd.Series([False] * len(df))).astype(bool)
    n_seen = int(seen.sum())
    audit = {
        "n_rows_seen_in_training": n_seen,
        "frac_rows_seen_in_training": float(seen.mean()),
        "n_rows_stimulus_overlap": int((~df.get(
            "stimulus_disjoint", pd.Series([True] * len(df))).astype(bool)).sum()),
        "n_train_subjects_values": sorted(int(v) for v in
                                          pd.unique(df["n_train_subjects"])),
        "subject_ids_in_rows": sorted(int(v) for v in pd.unique(df["subject_id"])),
        "n_subject_ids_in_rows": int(df["subject_id"].nunique()),
        "regimes_in_rows": sorted(set(df["regime"].astype(str)))
        if "regime" in df.columns else None,
        "standardize_in_rows": sorted(set(df["standardize"].astype(str)))
        if "standardize" in df.columns else None,
        "k_in_rows": sorted(int(v) for v in pd.unique(df["k"])) if "k" in df.columns else None,
    }
    if n_seen and args.regime != "within_subject":
        log.error(
            "CONTAMINATION: %d/%d rows have seen_in_training=True in a %s run "
            "(n_train_subjects seen in rows: %s, %d distinct query subjects). "
            "Cross-subject transfer is NOT what these rows measure. If this "
            "directory was reused, delete it and re-run with --force.",
            n_seen, len(df), args.regime, audit["n_train_subjects_values"],
            audit["n_subject_ids_in_rows"],
        )
    if len(audit["n_train_subjects_values"]) > 1:
        log.error("CONTAMINATION: rows in this directory come from runs with "
                  "different training-population sizes %s -- folds were mixed "
                  "across configurations.", audit["n_train_subjects_values"])

    # ---- SRM vs the no-SRM control, side by side ---------------------------- #
    n_test_cond = int(df["n_test_conditions"].iloc[0])
    chance = {
        "match/top1": 1.0 / max(1, n_test_cond),
        "video/top1": 1.0 / max(1, n_test_cond),
        "video_base/top1": (1.0 / int(df["n_test_videos"].iloc[0])
                            if "n_test_videos" in df.columns else None),
        "top1_g18": 1.0 / 18,
    }
    def _chance_of(metric: str) -> float:
        """1/gallery for the gallery this metric was actually scored against."""
        if metric.endswith("_g18"):
            return 1.0 / 18
        if metric.startswith("video_base/"):
            return (1.0 / int(df["n_test_videos"].iloc[0])
                    if "n_test_videos" in df.columns else float("nan"))
        return 1.0 / max(1, n_test_cond)

    full = curve.loc[curve["enroll_frac"].idxmax()]  # full-enrollment row
    vs: Dict[str, Any] = {}
    for m in ("match/top1", "match/top1_g18", "video/top1", "video/top1_g18",
              "video_base/top1", "video_base/top1_g18"):
        c = f"control_{m}"
        if m in curve.columns and c in curve.columns:
            vs[m] = {"srm": float(full[m]), "no_srm_control": float(full[c]),
                     "srm_minus_control": float(full[m]) - float(full[c]),
                     "chance": _chance_of(m)}
    srm_wins = [k for k, v in vs.items() if v["srm_minus_control"] > 0]
    atomic_write_json(out_dir / "summary.json", {
        "status": "ok",
        "n_rows": len(df), "n_folds": int(df["fold_index"].nunique()),
        "n_folds_skipped": n_skipped, "skip_reasons": skip_reasons,
        "stimulus_disjoint": bool(df.get("stimulus_disjoint", pd.Series([True])).all()),
        "config_hash": cfg_hash, "config": cfg,
        "audit": audit,
        "srm_objective_drop_mean": float(df["srm_objective_drop"].mean())
        if "srm_objective_drop" in df.columns else None,
        "worst_subject_objective_share_max": float(df["worst_subject_objective_share"].max())
        if "worst_subject_objective_share" in df.columns else None,
        "chance": chance,
        "srm_vs_control": vs,
        "srm_beats_control_on": srm_wins,
        # kept at the top level for the older readers that grep for them
        "regime": args.regime, "k": args.k, "feature_mode": args.feature_mode,
        "standardize": args.standardize,
        "subject_energy_norm": bool(args.subject_energy_norm),
        "model_tag": args.model_tag if args.to_video else None,
        "enrollment_curve": curve.to_dict("records"),
    })
    cols = [c for c in ("enroll_frac", "n_enroll_conditions", "seen_in_training",
                        "match/top1", "control_match/top1",
                        "match/top1_g18", "control_match/top1_g18",
                        "video_base/top1", "control_video_base/top1")
            if c in curve.columns]
    log.info("enrollment curve (control_* = same protocol, no SRM projection):\n%s",
             curve[cols].to_string(index=False))
    for m, v in vs.items():
        log.info("%-22s SRM %.4f | no-SRM control %.4f | delta %+.4f | chance %.4f",
                 m, v["srm"], v["no_srm_control"], v["srm_minus_control"], v["chance"])
    if vs and not srm_wins:
        log.warning("SRM does not beat the no-SRM control on any reported metric. "
                    "That is a legitimate result; report the pair, never the SRM "
                    "number alone.")
    log.info("wrote %s", out_dir / "summary.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
