"""Linear EEG -> video-embedding alignment (EA + ridge + cosine retrieval).

The control that decides whether the deep encoder earns its keep.  Flatten the
epoch into ``64 x T`` features, optionally Euclidean-align the subject, fit a
multi-output ridge onto the frozen video embedding, and retrieve by cosine
against the same galleries the deep model is scored on.  Same folds, same
gallery sizes, same pseudo-trial estimator -- the only thing that changes is the
map from EEG to embedding.

Two regimes, mirroring the main protocol:

* **within-subject** -- video-disjoint folds, every subject in both splits.
  ``--per-subject-model`` gives the literal "one ridge per subject" reading (the
  ceiling a linear map can reach when it gets to memorize one head); it is
  **not** the default, because on this dataset a fold leaves only ~1.4k training
  trials per subject against p=1920 features and the per-subject fit sits at
  chance for every alpha.  The default fits one ridge on the EA-whitened trials
  pooled over all training subjects and applies it to the same subjects' held-out
  videos.  Both are within-subject; say which one you ran.
* **cross-subject** (``loso`` / ``double_disjoint``) -- one ridge fitted on the
  pooled training subjects and applied to held-out subjects.  Without EA this is
  usually near chance, which is precisely the point: it isolates how much of the
  deep model's cross-subject transfer is attributable to whitening rather than
  to the learned encoder.

The penalty is chosen by generalized cross-validation *on the training fold*, in
closed form from one eigendecomposition, so the whole alpha path is nearly free
and the test fold is never consulted.

Reading the output
------------------
Folds come from :func:`tactus.data.splits.make_folds`, so they are base-video
disjoint by construction (``run_fold`` re-asserts it per fold and refuses to
score a fold whose train and test share a base video).  EA and the robust
per-channel scaling are fit on the *training* trials of each fold; only a test
subject with no training trials at all -- which never happens under
``within_subject`` -- falls back to the documented transductive fit, and the
count is reported as ``n_transductive_ea_subjects``.

Every retrieval block is emitted twice:

``{tag}/video/*``
    Protocol-identical -- cosine against the full ridge prediction, the same
    scorer the deep model is graded with.
``{tag}/video_nomean/*``
    The same scorer after removing the fitted training-mean embedding.  That
    term is a constant shared by every query, so it contributes a
    query-independent bias to each gallery item's score; the SigLIP gallery is
    anisotropic enough (mean pairwise cosine 0.88) that it swamps the
    EEG-dependent term and pins the headline number to chance.  Neither is
    silently preferred: report both and say which one the paper quotes.
``{tag}/video_centered/*``
    The convention that ships (DECISIONS D12): the training mean removed from
    the query *and* the gallery.  Defined once in the module docstring of
    :mod:`tactus.eval.retrieval` ("centring convention"); see there for why
    centring one side alone is a different estimator rather than a milder one.
    The paper quotes this -- it is :data:`PRIMARY_VARIANT`, and it is the only
    variant the uncertainty stage calls "the headline".  ``video`` and
    ``video_nomean`` stay in the output as the before/after the decision asked
    for, not as alternatives anyone may quote.

Uncertainty on the headline
---------------------------
``summary.json`` carries the same two uncertainty statements the deep baselines
carry, computed by the same shared code, so the three numbers are comparable:

* **95% CI** -- :func:`tactus.eval.stats.bootstrap_ci` over the *subject*, after
  averaging the folds within each subject (the convention in
  :func:`tactus.eval.report.aggregate_folds`).  Bootstrapping trials would be
  wrong: a subject's ~9k pseudo-trials are one cluster, not 9k units.
* **Permutation p** -- :func:`tactus.eval.permutation.video_level_permutation_test`
  with the **base video** as the exchangeable unit: the shuffle relabels the
  fold's held-out gallery videos, never the trials.  A trial-level null is also
  emitted (``trial_level_null_diagnostic``) but *only* to report the narrowing
  factor; it must never be quoted as a p-value.  A material-stratified variant
  answers the sharper question "is there video information beyond material?".

Attribute-shortcut control -- read this before quoting a number
---------------------------------------------------------------
The size-harmonised control for "is this just an 8-way material code?" is
**within-material 2-way top-1 vs overall 2-way top-1**.  Both are computed by
the shared scorer on the same queries and are reported side by side, together
with the surviving fraction ``(within - 0.5) / (overall - 0.5)``; at n=80 that
is 0.5722 vs 0.5946, i.e. ~76% of pairwise discriminability survives material
control.  ``material_control_ratio`` in each fold json is exactly this quantity.

``cross_group_*`` is **not** that control and earlier drafts of this module
misread it as one.  :func:`tactus.eval.retrieval.retrieval_metrics` builds the
cross-group gallery as *all other-material items plus the true item*, so the
true item is the only one of its material and a perfect material classifier
scores 100%.  The 0.1718 (chance 0.0683) that was once quoted as "+10.4 pts
after controlling for material" is an upper bound **inflated by** the material
code, i.e. the opposite of a control.  The keys are therefore suppressed by
default (``--keep-cross-group`` restores them for anyone who wants to audit the
claim); the bug was in this module's interpretation, not in ``retrieval.py``.

Protocol audit trail
--------------------
``--decim-mode`` changed from ``slice`` to ``pool`` (boxcar anti-alias) during
repair, and that is not cosmetic -- it moves the headline.  ``summary.json``
therefore carries a ``decim_mode_audit`` block that pairs this run's folds
against the sibling result directory that differs *only* in the decimation mode
and reports both headline numbers per fold plus their difference, so the choice
is auditable from the artifact instead of from a changelog.

Known inconsistency with ``srm.py`` (not fixed here)
----------------------------------------------------
``srm.py`` scores its ridge-to-video endpoint after removing the training mean
from **both** the prediction and the gallery; this module removes it from the
**query only**.  Two baselines for the same 18-way endpoint therefore use two
centring conventions, which changes the geometry of the cosine and is not a
relabelling.  It is recorded in ``summary.json`` under ``known_inconsistencies``
and needs a human decision; nothing here silently harmonises it.

    python -m tactus.baselines.linear_align --regime double_disjoint --ea
    python -m tactus.baselines.linear_align --regime within_subject --model-tag siglip2-base
    python -m tactus.baselines.linear_align --regime within_subject --subjects 1-3 --folds 0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from tactus.common import (
    EpochStore,
    SubjectTransform,
    VideoEmbeddings,
    atomic_write_json,
    attach_label_ids,
    fit_subject_transforms,
    hash_obj,
    load_trials,
    load_video_emb,
    results_dir,
)

log = logging.getLogger(__name__)

#: The variant the paper quotes as the linear-baseline headline (see the module
#: docstring): cosine after removing the fitted training-mean embedding from the
#: query.  The uncertainty stage is computed for both variants; only this one is
#: labelled "headline".
#: DECISIONS D12.  Was ``video_nomean`` (query centred, gallery not),
#: which is the mismatched convention that pinned the headline to chance.
PRIMARY_VARIANT = "video_centered"

#: Gallery size of the pre-registered primary endpoint (18-way top-1).
PRIMARY_NWAY = 18

__all__ = [
    "RidgeAlpha", "retrieval_np", "featurize", "run_fold", "parse_subjects",
    "PRIMARY_VARIANT", "PRIMARY_NWAY", "subject_bootstrap_ci",
    "video_permutation_test", "decim_mode_audit", "main",
]


# --------------------------------------------------------------------------- #
# ridge with closed-form GCV over the whole alpha path
# --------------------------------------------------------------------------- #


class RidgeAlpha:
    """Multi-output ridge whose penalty is picked by GCV on the training set.

    Solves in the primal (``p x p`` eigendecomposition) when ``p <= n`` and in
    the dual (``n x n``) otherwise, so it is efficient in both the
    "one subject, 7680 features" and the "40k pooled trials, 1920 features"
    corners.  One eigendecomposition serves every candidate ``alpha``:

        RSS(a) = ||Y||^2 - 2 * sum_j g_j^2/(l_j+a) + sum_j l_j g_j^2/(l_j+a)^2
        tr H(a) = sum_j l_j/(l_j+a)
        GCV(a) = n * RSS(a) / (n - tr H(a))^2

    Both X and Y are centered, and the intercept is restored at predict time.
    """

    def __init__(self, alphas: Sequence[float] = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)) -> None:
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.coef_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.alpha_: float = float("nan")
        self.gcv_: np.ndarray | None = None
        self.at_boundary_: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeAlpha":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = x.shape
        self.x_mean_ = x.mean(axis=0, keepdims=True)
        self.y_mean_ = y.mean(axis=0, keepdims=True)
        xc = x - self.x_mean_
        yc = y - self.y_mean_
        yy = float((yc ** 2).sum())

        if p <= n:  # primal
            lam, v = np.linalg.eigh(xc.T @ xc)
            g = v.T @ (xc.T @ yc)  # (p, D)
            lam = np.maximum(lam, 0.0)
            g2 = (g ** 2).sum(axis=1)  # (p,)
            rss = np.array([
                yy - 2 * float((g2 / (lam + a)).sum()) + float((lam * g2 / (lam + a) ** 2).sum())
                for a in self.alphas
            ])
            trh = np.array([float((lam / (lam + a)).sum()) for a in self.alphas])
            gcv = n * np.maximum(rss, 1e-30) / np.maximum(n - trh, 1e-6) ** 2
            a = float(self.alphas[int(np.argmin(gcv))])
            self.coef_ = v @ (g / (lam + a)[:, None])
        else:  # dual
            k = xc @ xc.T
            lam, u = np.linalg.eigh(k)
            lam = np.maximum(lam, 0.0)
            ut = u.T @ yc  # (n, D)
            u2 = (ut ** 2).sum(axis=1)
            rss = np.array([
                yy - 2 * float((lam * u2 / (lam + a)).sum())
                + float((lam ** 2 * u2 / (lam + a) ** 2).sum())
                for a in self.alphas
            ])
            trh = np.array([float((lam / (lam + a)).sum()) for a in self.alphas])
            gcv = n * np.maximum(rss, 1e-30) / np.maximum(n - trh, 1e-6) ** 2
            a = float(self.alphas[int(np.argmin(gcv))])
            self.coef_ = xc.T @ (u @ (ut / (lam + a)[:, None]))

        self.alpha_ = a
        self.gcv_ = gcv
        self.at_boundary_ = bool(
            self.alphas.size > 1
            and (a <= float(self.alphas.min()) or a >= float(self.alphas.max()))
        )
        if self.at_boundary_:
            # Not cosmetic.  On this dataset the GCV surface is almost flat above
            # ~1e4 and the argmin lands on the top of the grid, which shrinks the
            # fit to the intercept (in-sample R^2 ~0.005) and drives retrieval to
            # exactly chance.  Warn loudly instead of hiding it at DEBUG.
            log.warning(
                "GCV selected a boundary alpha (%.3g out of [%.3g, %.3g]) on n=%d, p=%d; "
                "the fit may be shrunk to the intercept -- widen --alphas or check "
                "'alpha_at_boundary' in the fold json",
                a, float(self.alphas.min()), float(self.alphas.max()), n, p,
            )
        return self

    def predict(self, x: np.ndarray, *, intercept: bool = True) -> np.ndarray:
        """Predicted embeddings.

        ``intercept=False`` returns only the EEG-dependent part
        ``(x - x_mean) @ coef``, i.e. the prediction with the fitted training
        mean embedding ``y_mean_`` removed.  That constant is identical for every
        query, so under cosine retrieval it contributes a query-independent bias
        ``y_mean . g_k`` to the score of gallery item ``k`` and can only blur the
        ranking; on this (highly anisotropic: mean pairwise cosine 0.88) SigLIP
        gallery it dominates the EEG-dependent term by ~8x and pins 18-way top-1
        to chance.  Both variants are scored and reported -- see ``run_fold``.
        """
        if self.coef_ is None:
            raise RuntimeError("RidgeAlpha.predict called before fit")
        z = (np.asarray(x, dtype=np.float64) - self.x_mean_) @ self.coef_
        return z + self.y_mean_ if intercept else z


# --------------------------------------------------------------------------- #
# retrieval (numpy; no torch in the baselines)
# --------------------------------------------------------------------------- #


def retrieval_np(
    z: np.ndarray,
    gallery: np.ndarray,
    true_idx: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    gallery_sizes: Sequence[int] = (2, 10, 18, 90),
    n_draws: int = 20,
    seed: int = 0,
    subject_id: np.ndarray | None = None,
    include_cross_group: bool = False,
) -> Dict[str, Any]:
    """Cosine retrieval metrics, full gallery plus random sub-galleries.

    Defers to ``tactus.eval.retrieval.retrieval_metrics`` for the full-gallery
    numbers when that module exists, so the baselines and the deep model are
    scored by the same code; the local implementation is the fallback.

    ``gallery_sizes`` is **forwarded** to the shared scorer.  It used not to be,
    so every ``nway{N}_*`` key -- including the primary ``nway18_top1`` -- came
    from ``DEFAULT_GALLERY_SIZES`` no matter what the caller asked for.  That was
    harmless only because the defaults happen to contain 18; it would have gone
    wrong silently the first time anyone passed ``--gallery-sizes``.

    ``include_cross_group`` defaults to **False**.  ``cross_group_*`` scores the
    true item against a gallery of items that are all of *other* materials, so
    the true item is the only one of its material and a pure material classifier
    scores 100%: it is an upper bound inflated by the attribute code, not a
    control for it.  The size-harmonised control is ``within_group_nway2_top1``
    vs ``nway2_top1``, which is always emitted.
    """
    z = _l2(np.asarray(z, dtype=np.float64))
    gallery = _l2(np.asarray(gallery, dtype=np.float64))
    true_idx = np.asarray(true_idx, dtype=np.int64)
    sim = z @ gallery.T
    true_score = sim[np.arange(sim.shape[0]), true_idx]
    rank = 1 + (sim > true_score[:, None]).sum(axis=1)

    out: Dict[str, Any] = {
        "n": int(sim.shape[0]), "gallery": int(gallery.shape[0]),
        "top1": float((rank == 1).mean()), "top5": float((rank <= 5).mean()),
        "mean_rank": float(rank.mean()), "median_rank": float(np.median(rank)),
        "percentile_rank": float((1 - (rank - 1) / max(1, gallery.shape[0] - 1)).mean()),
    }
    try:  # keep the deep model and the baselines on one scorer when possible
        from tactus.eval.retrieval import retrieval_metrics  # type: ignore
    except ImportError:  # pragma: no cover - the shared scorer is expected to exist
        retrieval_metrics = None  # type: ignore[assignment]
    ref = None
    if retrieval_metrics is not None:
        try:
            ref = retrieval_metrics(
                z.astype(np.float32), gallery.astype(np.float32), true_idx, groups,
                gallery_sizes=tuple(int(s) for s in gallery_sizes),
                include_cross_group=bool(include_cross_group),
            )
        except Exception:  # never silently swap scorers without saying so
            log.exception(
                "shared scorer tactus.eval.retrieval.retrieval_metrics failed; "
                "falling back to the local implementation -- the emitted keys are "
                "NOT the ones the deep model is graded with"
            )
            out["shared_scorer_failed"] = 1.0
    if ref is not None:
        out.update({k: float(v) for k, v in ref.items() if isinstance(v, (int, float))})
    elif groups is not None and np.asarray(groups).shape[0] == gallery.shape[0]:
        groups = np.asarray(groups)
        same = groups[None, :] == groups[true_idx][:, None]
        r = 1 + (np.where(same, sim, -np.inf) > true_score[:, None]).sum(axis=1)
        out["within_group_top1"] = float((r == 1).mean())

    rng = np.random.default_rng(seed)
    g = gallery.shape[0]
    for size in gallery_sizes:
        if size < 2 or size > g:
            continue
        if size == g:
            out[f"top1_g{int(size)}"] = out["top1"]
            continue
        t1 = []
        for _ in range(n_draws):
            t1.append(_draw_top1(sim, true_idx, true_score, int(size), rng))
        out[f"top1_g{int(size)}"] = float(np.mean(t1))

    if subject_id is not None:
        per: Dict[str, Dict[str, float]] = {}
        subject_id = np.asarray(subject_id)
        for s in np.unique(subject_id):
            m = subject_id == s
            per[str(int(s))] = {
                "top1": float((rank[m] == 1).mean()),
                "top5": float((rank[m] <= 5).mean()),
                "mean_rank": float(rank[m].mean()),
                "n": int(m.sum()),
            }
        out["per_subject"] = per
    return out


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def draw_distractors(
    n: int, g: int, size: int, true_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """``(n, size-1)`` distinct distractor columns per row, excluding the truth.

    Sampling **without** replacement is not a detail.  With replacement, an
    18-way gallery drawn from 72 candidates loses ~2 distractors to collisions,
    so the reported 18-way top-1 -- the pre-registered primary endpoint -- comes
    out systematically too high.  The random-key + ``argpartition`` trick gives
    a distinct sample per row in one vectorized pass.
    """
    keys = rng.random((n, g - 1), dtype=np.float32)
    cand = np.argpartition(keys, size - 2, axis=1)[:, : size - 1]
    return cand + (cand >= true_idx[:, None])  # skip over the true column


def _draw_top1(
    sim: np.ndarray, true_idx: np.ndarray, true_score: np.ndarray,
    size: int, rng: np.random.Generator,
) -> float:
    cand = draw_distractors(sim.shape[0], sim.shape[1], size, true_idx, rng)
    s = np.take_along_axis(sim, cand, axis=1)
    return float(((s > true_score[:, None]).sum(axis=1) == 0).mean())


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def featurize(
    trials: pd.DataFrame,
    store: EpochStore,
    transforms: Mapping[int, SubjectTransform],
    *,
    decim: int = 4,
    decim_mode: str = "pool",
    pseudo_k: int = 1,
    seed: int = 0,
) -> tuple:
    """Flatten epochs into ``(n, 64 * T//decim)`` design rows.

    With ``pseudo_k > 1`` the rows are disjoint k-repeat averages within each
    ``(subject, condition)`` group -- averaging the *signal*, not the features,
    because that is where the SNR gain lives and it is what the deep model's
    pseudo-trial evaluation does too.

    ``decim_mode`` controls how the time axis is shortened:

    ``"pool"`` (default)
        Non-overlapping boxcar average of ``decim`` samples.  The epochs are
        sampled at 200 Hz but only low-passed at 100 Hz, so plain subsampling
        by 4 folds everything between 25 and 100 Hz back into the retained band.
        The boxcar is the anti-alias filter that makes the decimation legal, and
        it costs nothing: same 1920 features, measurably better retrieval.
    ``"slice"``
        Plain ``x[..., ::decim]`` -- kept so the pre-fix behaviour can be
        reproduced exactly, not because it is correct.
    """
    if decim_mode not in ("pool", "slice"):
        raise ValueError(f"decim_mode must be 'pool' or 'slice', got {decim_mode!r}")
    rows: List[np.ndarray] = []
    metas: List[pd.DataFrame] = []
    rng = np.random.default_rng([seed, int(pseudo_k)])
    for sid, grp in trials.groupby("subject_id", sort=True):
        sid = int(sid)
        tf = transforms.get(sid)
        idx = grp["within_subj_idx"].to_numpy(np.int64)
        x = store.take(sid, idx)
        if tf is not None:
            x = tf.apply(x)
        if decim > 1:
            if decim_mode == "pool":
                n_keep = (x.shape[2] // decim) * decim
                x = x[:, :, :n_keep].reshape(
                    x.shape[0], x.shape[1], n_keep // decim, decim
                ).mean(axis=-1)
            else:
                x = x[:, :, ::decim]
        g = grp.reset_index(drop=True)
        if pseudo_k <= 1:
            rows.append(x.reshape(x.shape[0], -1))
            metas.append(g)
            continue
        chunks: List[np.ndarray] = []
        firsts: List[int] = []
        for _cond, cg in g.groupby("condition_id", sort=True):
            pos = rng.permutation(cg.index.to_numpy())
            n_full = pos.size // pseudo_k
            if n_full == 0:
                continue
            for chunk in np.split(pos[: n_full * pseudo_k], n_full):
                chunks.append(chunk)
                firsts.append(int(chunk[0]))
        if not chunks:
            continue
        rows.append(np.stack([x[c].mean(axis=0).reshape(-1) for c in chunks]))
        metas.append(g.iloc[firsts].reset_index(drop=True))
    if not rows:
        return np.zeros((0, 0), np.float32), pd.DataFrame()
    return (np.concatenate(rows, axis=0).astype(np.float32),
            pd.concat(metas, ignore_index=True))


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #


def _gallery_for(
    meta: pd.DataFrame, video: VideoEmbeddings, unit: str
) -> tuple:
    key = meta["video_id"].to_numpy(np.int64) if unit == "video" \
        else meta["condition_id"].to_numpy(np.int64)
    ids = np.unique(key)
    pos = {int(v): i for i, v in enumerate(ids)}
    true_idx = np.array([pos[int(v)] for v in key], dtype=np.int64)
    emb = video.base_emb[ids - 1] if unit == "video" else video.cond_emb[ids]
    groups = None
    if unit == "video" and "material_id" in meta.columns:
        m = dict(zip(meta["video_id"].astype(int), meta["material_id"].astype(int)))
        groups = np.array([m.get(int(i), -1) for i in ids])
    return emb, true_idx, groups, ids


def run_fold(
    fold: Any,
    trials: pd.DataFrame,
    store: EpochStore,
    video: VideoEmbeddings,
    *,
    ea: bool = True,
    decim: int = 4,
    decim_mode: str = "pool",
    per_subject_model: bool = False,
    max_train_trials: int | None = 40000,
    pseudo_k: int = 4,
    alphas: Sequence[float] = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
    gallery_sizes: Sequence[int] = (2, 10, 18, 90),
    units: Sequence[str] = ("video", "condition"),
    seed: int = 0,
    include_cross_group: bool = False,
    capture: MutableMapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fit and score one fold.  Returns a metrics dict.

    ``capture`` (optional) receives the per-query **rank matrix** of the primary
    endpoint -- one row per pseudo-trial, one column per held-out gallery video,
    holding the rank that gallery item takes for that query.  Everything the
    uncertainty stage needs (per-subject accuracy, the video-level permutation
    null, the trial-level narrowing diagnostic) is a gather out of that matrix,
    so the CI and the p-value are computed from the *same* ranks the headline is
    computed from rather than from a second, possibly-drifting scoring path.
    """
    by_uid = trials.set_index("trial_uid", drop=False)
    # reindex introduces NaN rows for uids the trial table no longer has (e.g.
    # post-target exclusions), which silently upcasts the integer columns
    tr = by_uid.reindex(np.asarray(fold.train_uid, np.int64)).dropna(
        subset=["subject_id"]).reset_index(drop=True)
    te = by_uid.reindex(np.asarray(fold.test_uid, np.int64)).dropna(
        subset=["subject_id"]).reset_index(drop=True)
    for df in (tr, te):
        for c in ("subject_id", "within_subj_idx", "video_id", "condition_id", "material_id"):
            if c in df.columns:
                df[c] = df[c].astype(np.int64)

    # EA + robust scaling are fit on TRAIN trials only.  ``held_out`` is the set
    # of test subjects with no training trials at all (loso / double_disjoint);
    # those get the documented transductive fit on their own trials.  Under
    # ``within_subject`` every subject is in train, so this stays empty -- the
    # count is recorded so the audit does not rest on that assumption.
    tf = fit_subject_transforms(tr, store, ea=ea, robust_scale=True, seed=seed)
    held_out = sorted(set(te["subject_id"].unique()) - set(tf))
    if held_out:
        tf.update(fit_subject_transforms(
            te.loc[te["subject_id"].isin(held_out)], store, ea=ea, robust_scale=True, seed=seed
        ))

    out: Dict[str, Any] = {
        "video_fold_id": int(getattr(fold, "video_fold_id", -1)),
        "subject_fold_id": (None if getattr(fold, "subject_fold_id", None) is None
                            else int(fold.subject_fold_id)),
        "regime": str(getattr(fold, "regime", "")),
        "ea": bool(ea), "decim": int(decim), "decim_mode": str(decim_mode),
        "per_subject_model": bool(per_subject_model),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_train_subjects": int(tr["subject_id"].nunique()),
        "n_test_videos": int(te["video_id"].nunique()),
        "n_train_videos": int(tr["video_id"].nunique()),
        "n_video_overlap_train_test": int(
            np.intersect1d(tr["video_id"].to_numpy(), te["video_id"].to_numpy()).size
        ),
        "n_transductive_ea_subjects": int(len(held_out)),
    }
    # ``loso`` is subject-disjoint only -- all 90 videos appear in both splits by
    # design, so the video-disjointness assertion applies to the other two
    # regimes.  Never let a zero-shot claim rest on an unchecked fold.
    if out["regime"] in ("within_subject", "double_disjoint") \
            and out["n_video_overlap_train_test"]:
        raise AssertionError(
            f"fold {out['video_fold_id']}: {out['n_video_overlap_train_test']} base "
            f"video(s) appear in both train and test under regime "
            f"{out['regime']!r} -- the folds are not video-disjoint and every "
            "retrieval number below would be leakage."
        )

    # --- D12 centring constants, from the training split only ---------------
    # Trial-weighted so this matches the ridge's own y_mean_ under the pooled
    # model; under per_subject_model the per-subject means differ only by trial
    # counts, and one fold-level centre keeps query and gallery on the same
    # origin (a per-subject gallery centre is not expressible in one array).
    y_center = video.cond_emb[tr["condition_id"].to_numpy(np.int64)].mean(
        axis=0, keepdims=True)
    base_center = video.base_emb[
        np.unique(tr["video_id"].to_numpy(np.int64)) - 1].mean(axis=0, keepdims=True)

    # The training design does not depend on the pseudo-trial factor k, so the
    # (expensive) featurize + eigendecomposition is done once per model scope and
    # reused for every k instead of being redone per k.
    _models: Dict[Any, RidgeAlpha] = {}

    def _fit(tr_df: pd.DataFrame, scope: Any) -> RidgeAlpha:
        model = _models.get(scope)
        if model is not None:
            return model
        xtr, mtr = featurize(tr_df, store, tf, decim=decim, decim_mode=decim_mode, seed=seed)
        if max_train_trials and xtr.shape[0] > max_train_trials:
            rng = np.random.default_rng([seed, 31])
            keep = np.sort(rng.choice(xtr.shape[0], size=int(max_train_trials), replace=False))
            xtr, mtr = xtr[keep], mtr.iloc[keep].reset_index(drop=True)
        ytr = video.cond_emb[mtr["condition_id"].to_numpy(np.int64)]
        model = RidgeAlpha(alphas).fit(xtr, ytr)
        model.n_fit_ = int(xtr.shape[0])  # type: ignore[attr-defined]
        _models[scope] = model
        return model

    def _fit_predict(tr_df: pd.DataFrame, te_df: pd.DataFrame, k: int, scope: Any) -> tuple:
        model = _fit(tr_df, scope)
        xte, mte = featurize(te_df, store, tf, decim=decim, decim_mode=decim_mode,
                             pseudo_k=k, seed=seed)
        z = model.predict(xte, intercept=False)  # EEG-dependent part only
        return model, z + model.y_mean_, z, mte

    for k in ([1, pseudo_k] if pseudo_k > 1 else [1]):
        tag = "single" if k == 1 else f"pseudo_k{k}"
        t0 = time.time()
        if per_subject_model:
            preds, preds_nc, metas, alphas_used, n_bnd = [], [], [], [], 0
            for sid in sorted(te["subject_id"].unique()):
                tr_s = tr.loc[tr["subject_id"] == sid]
                te_s = te.loc[te["subject_id"] == sid]
                if len(tr_s) < 50 or len(te_s) == 0:
                    continue
                model, p, p_nc, m = _fit_predict(tr_s, te_s, k, int(sid))
                preds.append(p)
                preds_nc.append(p_nc)
                metas.append(m)
                alphas_used.append(model.alpha_)
                n_bnd += int(model.at_boundary_)
            if not preds:
                continue
            pred = np.concatenate(preds)
            pred_nc = np.concatenate(preds_nc)
            mte = pd.concat(metas, ignore_index=True)
            out[f"{tag}/alpha_median"] = float(np.median(alphas_used))
            out[f"{tag}/alpha_at_boundary_frac"] = float(n_bnd / len(alphas_used))
        else:
            model, pred, pred_nc, mte = _fit_predict(tr, te, k, "pooled")
            out[f"{tag}/alpha"] = float(model.alpha_)
            out[f"{tag}/alpha_at_boundary"] = bool(model.at_boundary_)
            out[f"{tag}/n_ridge_fit"] = int(getattr(model, "n_fit_", 0))

        # ``{unit}``      -- protocol-identical: cosine on the full ridge
        #                    prediction, exactly the scorer the deep model uses.
        # ``{unit}_nomean`` -- the same scorer applied after removing the fitted
        #                    train-mean embedding, which is a query-independent
        #                    constant.  On this gallery it is ~8x larger than the
        #                    EEG-dependent term and pins the headline number to
        #                    chance, so both are reported rather than one being
        #                    quietly preferred.  See ``RidgeAlpha.predict``.
        for unit in units:
            gal, true_idx, groups, ids = _gallery_for(mte, video, unit)
            # DECISIONS D12 picked SRM's convention: the training mean comes off
            # the query AND the gallery.  It is defined once, in the module
            # docstring of tactus.eval.retrieval; every arm points there.  Each
            # side is centred by the training mean *of its own space*, which is
            # why the video-unit gallery uses base_center and not y_center -- the
            # ridge predicts into condition-embedding space, the 18-way gallery
            # lives in base-video space, and one mean cannot serve both.
            gal_c = gal - (base_center if unit == "video" else y_center)
            for suffix, z, gsel in ((unit, pred, gal),
                                    (f"{unit}_nomean", pred_nc, gal),
                                    (f"{unit}_centered", pred - y_center, gal_c)):
                m = retrieval_np(
                    z, gsel, true_idx, groups=groups, gallery_sizes=gallery_sizes,
                    seed=seed, subject_id=mte["subject_id"].to_numpy(),
                    include_cross_group=include_cross_group,
                )
                per_sub = m.pop("per_subject", {})
                out.update({f"{tag}/{suffix}/{key}": v for key, v in m.items()})
                out[f"{tag}/{suffix}/per_subject"] = per_sub
                # The size-harmonised attribute-shortcut control: how much of the
                # 2-way discriminability survives when both candidates are of the
                # same material.  1.0 = material contributes nothing; 0.0 = the
                # endpoint is a repackaged material code.  (``cross_group_*`` is
                # NOT this control -- see the module docstring.)
                ov, wi = m.get("nway2_top1"), m.get("within_group_nway2_top1")
                if ov is not None and wi is not None:
                    out[f"{tag}/{suffix}/material_control_ratio"] = \
                        _surviving_fraction(float(ov), float(wi))
                if capture is not None and unit == "video" and k == pseudo_k:
                    capture[f"rank_{suffix}"] = _rank_matrix(z, gal).astype(np.float32)
                    capture["true_idx"] = true_idx.astype(np.int32)
                    capture["subject_id"] = mte["subject_id"].to_numpy(np.int32)
                    capture["gallery_video_id"] = np.asarray(ids, np.int64)
                    capture["gallery_material"] = (
                        np.asarray(groups, np.int64) if groups is not None
                        else np.full(len(ids), -1, np.int64)
                    )
        out[f"{tag}/seconds"] = time.time() - t0
    return out


#: Smallest overall 2-way effect (above the 0.5 chance line) for which the
#: material-control ratio is reported.  Below it the denominator is noise and
#: the ratio explodes to meaningless values, so it is reported as NaN instead.
MIN_NWAY2_EFFECT = 0.01


def _surviving_fraction(overall_nway2: float, within_nway2: float) -> float:
    """Share of 2-way discriminability that survives material control.

    ``(within - 0.5) / (overall - 0.5)``.  Undefined (NaN) when the overall
    2-way effect is not at least :data:`MIN_NWAY2_EFFECT` above chance -- a ratio
    of two near-zero numbers is not a control, it is a random number.
    """
    denom = float(overall_nway2) - 0.5
    if not np.isfinite(denom) or denom < MIN_NWAY2_EFFECT:
        return float("nan")
    return float((float(within_nway2) - 0.5) / denom)


def _rank_matrix(z: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """``(n_query, n_gallery)`` matrix of the rank each gallery item takes.

    Column ``j`` is what :func:`tactus.eval.retrieval.rank_of_true` would return
    if item ``j`` were the true one, so the observed ranks are
    ``R[arange(n), true_idx]`` and the ranks under a gallery permutation are
    ``R[arange(n), perm[true_idx]]`` -- no rescoring, and the same tie
    convention (ties count as half) as the headline metric.

    The float32 round-trip is deliberate: :func:`retrieval_np` hands the shared
    scorer float32 arrays, so scoring here in full float64 would put the CI and
    the permutation test on a marginally different similarity matrix from the
    headline they are supposed to quantify.  ``main`` asserts the two agree.
    """
    from tactus.eval.retrieval import rank_of_true

    zq = _l2(np.asarray(z, np.float64)).astype(np.float32).astype(np.float64)
    gq = _l2(np.asarray(gallery, np.float64)).astype(np.float32).astype(np.float64)
    sim = zq @ gq.T
    n, m = sim.shape
    out = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        out[:, j] = rank_of_true(sim, np.full(n, j, dtype=np.int64))
    return out


# --------------------------------------------------------------------------- #
# uncertainty on the headline: subject bootstrap CI + video-level permutation
# --------------------------------------------------------------------------- #


@dataclass
class PrimaryFold:
    """One fold's rank matrix for the primary endpoint (see :func:`_rank_matrix`)."""

    fold_index: int
    ranks: np.ndarray          # (n_query, n_gallery)
    true_idx: np.ndarray       # (n_query,)
    subject_id: np.ndarray     # (n_query,)
    gallery_video_id: np.ndarray   # (n_gallery,)
    gallery_material: np.ndarray   # (n_gallery,)

    @property
    def n_items(self) -> int:
        return int(self.ranks.shape[1])

    def true_ranks(self, perm: np.ndarray | None = None) -> np.ndarray:
        """Ranks of the true item, optionally after relabelling the gallery."""
        col = self.true_idx if perm is None else np.asarray(perm)[self.true_idx]
        return self.ranks[np.arange(self.ranks.shape[0]), col]


def _top1(ranks: np.ndarray, n_items: int, n_way: int) -> float:
    """The endpoint statistic, computed by the shared scorer's own function."""
    from tactus.eval.retrieval import nway_topk_accuracy

    if ranks.size == 0:
        return float("nan")
    return float(nway_topk_accuracy(ranks, n_items, n_way, 1))


def subject_bootstrap_ci(
    folds: Sequence[PrimaryFold],
    *,
    n_way: int = PRIMARY_NWAY,
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """95% CI on the endpoint with the **subject** as the unit of inference.

    Folds are averaged within a subject first and the bootstrap resamples
    subjects, exactly as :func:`tactus.eval.report.aggregate_folds` does for the
    deep baselines.  Resampling trials instead would treat one subject's ~9k
    pseudo-trials as 9k independent units and produce an interval several times
    too narrow.
    """
    from tactus.eval.stats import bootstrap_ci

    per: Dict[int, List[float]] = {}
    for f in folds:
        for s in np.unique(f.subject_id):
            sel = f.subject_id == s
            per.setdefault(int(s), []).append(
                _top1(f.true_ranks()[sel], f.n_items, n_way)
            )
    if not per:
        return {"estimate": float("nan"), "n_subjects": 0}
    subject_ids = sorted(per)
    values = np.array([float(np.mean(per[s])) for s in subject_ids])
    boot = bootstrap_ci(values, n_boot=n_boot, ci=ci, seed=seed)
    pooled = [_top1(f.true_ranks(), f.n_items, n_way) for f in folds]
    return {
        "estimate": boot["estimate"],
        "ci_lo": boot["lo"],
        "ci_hi": boot["hi"],
        "se": boot["se"],
        "n_subjects": int(boot["n"]),
        "n_folds": len(folds),
        "unit": "subject",
        "n_boot": int(n_boot),
        "ci": float(ci),
        "method": ("folds averaged within subject, then percentile bootstrap "
                   "over subjects (tactus.eval.stats.bootstrap_ci)"),
        # The bootstrap point estimate is the unweighted mean over subjects; the
        # quoted headline is the unweighted mean over folds of the pooled
        # accuracy.  They differ only through unequal trial counts per subject,
        # but they are different estimators and both are recorded.
        "pooled_fold_mean": float(np.mean(pooled)),
        "pooled_fold_sd": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else float("nan"),
        "per_fold": {int(f.fold_index): v for f, v in zip(folds, pooled)},
        "subject_values": {int(s): float(np.mean(per[s])) for s in subject_ids},
    }


def video_permutation_test(
    folds: Sequence[PrimaryFold],
    *,
    n_way: int = PRIMARY_NWAY,
    n_perm: int = 1000,
    seed: int = 0,
    material_matched: bool = True,
    also_trial_level: bool = True,
) -> Dict[str, Any]:
    """Permutation p for the endpoint, exchangeable unit = the **base video**.

    The shuffle relabels the fold's held-out gallery videos (the same convention
    as :func:`tactus.eval.run_report.primary_permutation`, which applies one
    index permutation to every fold's gallery).  Trials, subjects and the EEG
    side are untouched, so anything the ridge learned from subject identity or
    trial position survives the shuffle: the null asks specifically whether
    *stimulus identity* is recovered.

    ``material_matched`` adds the stratified null -- videos are only exchanged
    with videos of the same material, drawn independently per fold because the
    materials sit at different gallery positions in different folds -- which is
    the sharper "beyond the attribute code" question.
    """
    from tactus.eval.permutation import (
        null_narrowing_report,
        permutation_pvalue,
        sample_base_permutation,
        trial_level_null_diagnostic,
        video_level_permutation_test,
    )

    if not folds:
        return {}
    sizes = {f.n_items for f in folds}
    stat_name = f"{PRIMARY_VARIANT}/nway{n_way}_top1"

    def stat(perm: np.ndarray) -> float:
        return float(np.mean([
            _top1(f.true_ranks(perm), f.n_items, n_way) for f in folds
        ]))

    out: Dict[str, Any] = {}
    if len(sizes) != 1:
        out["skipped"] = (f"folds have different gallery sizes {sorted(sizes)}; "
                          "one shared video permutation is not defined")
        return out
    n_base = sizes.pop()
    video = video_level_permutation_test(
        stat, n_base=n_base, n_perm=n_perm, seed=seed, statistic_name=stat_name,
    )
    out["video_level"] = video.summary()

    if material_matched:
        rng = np.random.default_rng([seed, 917])
        obs = video.observed
        null = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            null[i] = float(np.mean([
                _top1(f.true_ranks(sample_base_permutation(
                    f.n_items, rng=rng, strata=f.gallery_material)),
                    f.n_items, n_way)
                for f in folds
            ]))
        finite = null[np.isfinite(null)]
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        out["video_level_material_matched"] = {
            "statistic": stat_name,
            "observed": obs,
            "p_value": permutation_pvalue(obs, null, "greater"),
            "null_mean": float(np.mean(finite)) if finite.size else float("nan"),
            "null_sd": sd,
            "null_p95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
            "z_score": (float((obs - np.mean(finite)) / sd)
                        if sd and np.isfinite(sd) and sd > 0 else float("nan")),
            "n_perm": int(n_perm),
            "exchangeable_unit": "base_video (within material)",
            "n_units": int(n_base),
            "tail": "greater",
            "strata": "material, resampled independently per fold",
            # a material represented by a single held-out video cannot move, so
            # those videos are effectively fixed and pull the null toward the
            # observed value; the count says how much of the gallery that is
            "n_singleton_strata_per_fold": {
                int(f.fold_index): int(sum(
                    1 for _, c in zip(*np.unique(f.gallery_material,
                                                 return_counts=True)) if c == 1))
                for f in folds
            },
        }

    if also_trial_level:
        n_total = int(sum(f.ranks.shape[0] for f in folds))

        def stat_trial(perm_trials: np.ndarray) -> float:
            accs, off = [], 0
            for f in folds:
                n = f.ranks.shape[0]
                take = perm_trials[off:off + n] % n
                off += n
                r = f.ranks[np.arange(n), f.true_idx[take]]
                accs.append(_top1(r, f.n_items, n_way))
            return float(np.mean(accs))

        trial = trial_level_null_diagnostic(
            stat_trial, n_total, n_perm=n_perm, seed=seed + 1,
            observed=video.observed, statistic_name=stat_name,
        )
        out["trial_level_INVALID_diagnostic"] = trial.summary()
        out["narrowing"] = null_narrowing_report(video, trial)
    return out


# --------------------------------------------------------------------------- #
# protocol audit: the decim-mode A/B that moved the headline
# --------------------------------------------------------------------------- #
_DECIM_TAG_RE = re.compile(r"_d(\d+)([ps])")


def decim_mode_audit(out_root: Path, tag: str, headline_key: str) -> Dict[str, Any]:
    """Pair this run against the sibling run that differs only in ``--decim-mode``.

    The switch from ``slice`` to ``pool`` was a protocol change that moves the
    headline (up, i.e. in the conservative direction for a baseline), and until
    now nothing on disk recorded the size of the move.  This reads the *fold*
    jsons of both directories, intersects them on ``fold_index`` so the
    comparison is fold-matched, and reports both headline numbers plus the
    difference.  Absent sibling -> ``available: false`` and a note, never a
    silently missing field.
    """
    # last match: the decim field is appended after the model tag, so anchoring on
    # the end cannot be fooled by a model tag that happens to contain "_d<N><p|s>"
    hits = list(_DECIM_TAG_RE.finditer(tag))
    m = hits[-1] if hits else None
    audit: Dict[str, Any] = {
        "chosen": None if m is None else {"p": "pool", "s": "slice"}[m.group(2)],
        "headline_key": headline_key,
        "why": ("epochs are sampled at 200 Hz but low-passed at 100 Hz, so plain "
                "subsampling by 4 aliases 25-100 Hz back into the retained band; "
                "'pool' boxcar-averages first, which is the anti-alias filter that "
                "makes the decimation legal"),
        "available": False,
    }
    if m is None:
        audit["note"] = f"tag {tag!r} carries no _d<N><p|s> field; nothing to pair"
        return audit
    other = {"p": "s", "s": "p"}[m.group(2)]
    sib_tag = tag[: m.start(2)] + other + tag[m.end(2):]
    here, there = out_root, out_root.parent / sib_tag
    audit["runs"] = {audit["chosen"]: str(here),
                     {"p": "pool", "s": "slice"}[other]: str(there)}
    if not (there / "folds").is_dir():
        audit["note"] = (f"no sibling run at {there} -- run the same command with "
                         f"--decim-mode {'slice' if other == 's' else 'pool'} to "
                         "make the comparison auditable")
        return audit

    def _folds(root: Path) -> Dict[int, float]:
        vals: Dict[int, float] = {}
        for fp in sorted((root / "folds").glob("fold*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if headline_key in d and "fold_index" in d:
                vals[int(d["fold_index"])] = float(d[headline_key])
        return vals

    a, b = _folds(here), _folds(there)
    common = sorted(set(a) & set(b))
    if not common:
        audit["note"] = (f"sibling {sib_tag} exists but shares no fold_index with "
                         "this run; nothing fold-matched to compare")
        return audit
    mode_here = audit["chosen"]
    mode_there = {"p": "pool", "s": "slice"}[other]
    mean_here = float(np.mean([a[i] for i in common]))
    mean_there = float(np.mean([b[i] for i in common]))
    audit.update({
        "available": True,
        "n_folds_compared": len(common),
        "folds_compared": common,
        mode_here: mean_here,
        mode_there: mean_there,
        "per_fold": {str(i): {mode_here: a[i], mode_there: b[i]} for i in common},
        "delta_pool_minus_slice": float(
            (mean_here - mean_there) if mode_here == "pool" else (mean_there - mean_here)
        ),
    })
    return audit


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tactus.baselines.linear_align",
        description="Ridge (EEG -> frozen video embedding) + cosine retrieval baseline.",
    )
    p.add_argument("--regime", default="double_disjoint",
                   choices=["within_subject", "loso", "double_disjoint"])
    p.add_argument("--window", default="w0600")
    p.add_argument("--model-tag", default="siglip2-base",
                   help="video-embedding cache stem under data/derived/video_emb/")
    p.add_argument("--subjects", default=None,
                   help="restrict to these subjects, e.g. '1-3' or '1,5,9' "
                        "(smoke tests only; results land in a separate tag)")
    p.add_argument("--ea", dest="ea", action="store_true", default=True,
                   help="Euclidean-align each subject (default on)")
    p.add_argument("--no-ea", dest="ea", action="store_false")
    p.add_argument("--decim", type=int, default=4,
                   help="temporal decimation; 4 gives 64x30=1920 features")
    p.add_argument("--decim-mode", default="pool", choices=["pool", "slice"],
                   help="'pool' = boxcar-average decim samples (anti-aliased; the "
                        "epochs are 200 Hz but only low-passed at 100 Hz, so plain "
                        "subsampling folds 25-100 Hz back in). 'slice' reproduces "
                        "the pre-fix x[..., ::decim] behaviour.")
    p.add_argument("--per-subject-model", action="store_true",
                   help="one ridge per subject (the within-subject ceiling)")
    p.add_argument("--n-video-folds", type=int, default=5)
    p.add_argument("--n-subject-folds", type=int, default=8)
    p.add_argument("--folds", type=lambda s: [int(x) for x in s.split(",") if x], default=None)
    p.add_argument("--max-train-trials", type=int, default=40000)
    p.add_argument("--pseudo-k", type=int, default=4)
    p.add_argument("--alphas", default="1,10,100,1000,10000,100000,1000000")
    p.add_argument("--gallery-sizes", default="2,10,18,90")
    p.add_argument("--keep-cross-group", action="store_true",
                   help="re-emit the cross_group_* keys.  They are OFF by default: "
                        "the cross-group gallery holds every other-material item "
                        "plus the true one, so a pure material classifier scores "
                        "100% and the number is an upper bound INFLATED by the "
                        "attribute code, not a control for it.  The control is "
                        "within_group_nway2_top1 vs nway2_top1.")
    p.add_argument("--n-boot", type=int, default=10000,
                   help="bootstrap resamples for the subject-level 95%% CI")
    p.add_argument("--n-perm", type=int, default=1000,
                   help="permutations for the video-level null (unit = base video)")
    p.add_argument("--skip-stats", action="store_true",
                   help="skip the CI / permutation stage (fold metrics only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def parse_subjects(spec: str | None) -> List[int] | None:
    """``"1-3,7"`` -> ``[1, 2, 3, 7]``; ``None`` -> ``None`` (all subjects)."""
    if spec is None or not str(spec).strip():
        return None
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )
    from tactus.data.splits import make_folds

    subjects = parse_subjects(args.subjects)

    out_root = Path(args.out) if args.out else results_dir() / "baselines" / "linear_align"
    tag = (f"{args.regime}_{args.window}_{args.model_tag}"
           f"_ea{int(args.ea)}_d{args.decim}{args.decim_mode[0]}")
    if args.per_subject_model:
        tag += "_persub"
    if subjects is not None:  # never collide with the full-cohort results
        tag += f"_s{len(subjects)}of{hash_obj(subjects, 6)}"
    out_root = out_root / tag
    (out_root / "folds").mkdir(parents=True, exist_ok=True)

    trials = attach_label_ids(load_trials(subjects=subjects))
    store = EpochStore(args.window)
    video = load_video_emb(args.model_tag)
    folds = make_folds(trials, args.regime, n_video_folds=args.n_video_folds,
                       n_subject_folds=args.n_subject_folds, seed=args.seed,
                       subjects=subjects)
    log.info("%s: %d folds over %d subjects / %d videos",
             args.regime, len(folds), trials["subject_id"].nunique(),
             trials["video_id"].nunique())

    alphas = [float(a) for a in args.alphas.split(",")]
    sizes = [int(s) for s in args.gallery_sizes.split(",")]
    if 2 not in sizes:
        # the size-harmonised attribute-shortcut control is the 2-way pair
        # (within-material vs overall); without size 2 the shared scorer never
        # emits it and the run would ship without its shortcut control
        sizes = sorted(set(sizes) | {2})
        log.info("added gallery size 2: it carries the material-shortcut control "
                 "(within_group_nway2_top1 vs nway2_top1)")
    rows: List[Dict[str, Any]] = []
    for i, fold in enumerate(folds):
        if args.folds and i not in args.folds:
            continue
        key = f"fold{i:03d}"
        fpath = out_root / "folds" / f"{key}.json"
        cpath = out_root / "folds" / f"{key}_primary.npz"
        if fpath.exists() and not args.force:
            with open(fpath, "r", encoding="utf-8") as fh:
                rows.append(json.load(fh))
            log.info("%s: cached", key)
            continue
        t0 = time.time()
        capture: Dict[str, Any] = {}
        res = run_fold(
            fold, trials, store, video, ea=args.ea, decim=args.decim,
            decim_mode=args.decim_mode, per_subject_model=args.per_subject_model,
            max_train_trials=args.max_train_trials, pseudo_k=args.pseudo_k,
            alphas=alphas, gallery_sizes=sizes, seed=args.seed,
            include_cross_group=args.keep_cross_group, capture=capture,
        )
        res["fold_index"] = i
        atomic_write_json(fpath, res)
        if capture:
            np.savez_compressed(cpath, fold_index=np.int64(i), **capture)
        rows.append(res)
        # run_fold tags the k=1 block "single", so pseudo_k=1 has no "pseudo_k1"
        pk = f"pseudo_k{args.pseudo_k}" if args.pseudo_k > 1 else "single"
        log.info(
            "%s: 18-way top1 video | single %.4f (nomean %.4f)  %s %.4f (nomean %.4f)"
            "  [chance %.4f]  (%.1f min)",
            key,
            res.get("single/video/nway18_top1", float("nan")),
            res.get("single/video_nomean/nway18_top1", float("nan")),
            pk,
            res.get(f"{pk}/video/nway18_top1", float("nan")),
            res.get(f"{pk}/video_nomean/nway18_top1", float("nan")),
            res.get("single/video/nway18_chance_top1", float("nan")),
            (time.time() - t0) / 60,
        )

    if rows:
        flat = pd.DataFrame([
            {k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows
        ])
        flat.to_csv(out_root / "folds.csv", index=False)
        summary = {c: float(flat[c].mean()) for c in flat.columns
                   if flat[c].dtype.kind in "fiub"}
        # run_fold tags the k=1 block "single", so pseudo_k=1 has no "pseudo_k1"
        pk = f"pseudo_k{args.pseudo_k}" if args.pseudo_k > 1 else "single"
        headline_key = f"{pk}/{PRIMARY_VARIANT}/nway{PRIMARY_NWAY}_top1"
        doc: Dict[str, Any] = {
            "tag": tag,
            "n_folds": len(rows),
            "config": {
                "regime": args.regime, "window": args.window,
                "model_tag": args.model_tag, "ea": bool(args.ea),
                "decim": args.decim, "decim_mode": args.decim_mode,
                "per_subject_model": bool(args.per_subject_model),
                "pseudo_k": args.pseudo_k, "max_train_trials": args.max_train_trials,
                "alphas": alphas, "seed": args.seed, "gallery_sizes": sizes,
                "keep_cross_group": bool(args.keep_cross_group),
                "subjects": subjects, "folds_from": "tactus.data.splits.make_folds",
            },
            "means": summary,
        }
        doc["headline"] = _headline_block(summary, headline_key)
        doc["attribute_shortcut_control"] = _shortcut_block(summary, pk)
        doc["decim_mode_audit"] = decim_mode_audit(out_root, tag, headline_key)
        doc["known_inconsistencies"] = []
        doc["resolved_inconsistencies"] = [
            {
                "id": "centring_convention_vs_srm",
                "where": "tactus/baselines/srm.py vs tactus/baselines/linear_align.py",
                "what": ("srm.py subtracted the training mean embedding from the "
                         "prediction AND the gallery (y_center for the condition "
                         "gallery, base_center for the 18-way base-video gallery); "
                         "linear_align subtracted the ridge's y_mean_ from the query "
                         "only and left the gallery untouched."),
                "resolution": ("DECISIONS D12 adopted srm.py's convention everywhere. "
                               "linear_align now emits {unit}_centered and quotes it "
                               "as PRIMARY_VARIANT; the convention is defined once in "
                               "the module docstring of tactus.eval.retrieval."),
                "before_after": ("the superseded variants stay in this file's output "
                                 "as video (no centring) and video_nomean (query only) "
                                 "so the effect of the decision is auditable, not "
                                 "asserted"),
                "status": "fixed",
            },
        ]
        if not args.skip_stats:
            stats = _uncertainty(out_root, rows, args, headline_key)
            if stats:
                doc["headline"].update(stats)
        atomic_write_json(out_root / "summary.json", doc)
        log.info(
            "SUMMARY over %d fold(s): 18-way top1 video  single %.4f (nomean %.4f)"
            " | %s %.4f (nomean %.4f)  [chance %.4f]",
            len(rows),
            summary.get("single/video/nway18_top1", float("nan")),
            summary.get("single/video_nomean/nway18_top1", float("nan")),
            pk,
            summary.get(f"{pk}/video/nway18_top1", float("nan")),
            summary.get(f"{pk}/video_nomean/nway18_top1", float("nan")),
            summary.get("single/video/nway18_chance_top1", float("nan")),
        )
        _log_headline(doc["headline"], doc["attribute_shortcut_control"],
                      doc["decim_mode_audit"])
        log.info("wrote %s", out_root / "summary.json")
    return 0


def _headline_block(summary: Mapping[str, float], headline_key: str) -> Dict[str, Any]:
    return {
        "key": headline_key,
        "variant": PRIMARY_VARIANT,
        "variant_note": ("cosine after removing the fitted train-mean embedding "
                         "from the QUERY only; the with-mean variant is emitted "
                         "next to it as '%s'" % headline_key.replace("_nomean", "")),
        "fold_mean": summary.get(headline_key, float("nan")),
        "chance": summary.get(headline_key.replace("_top1", "_chance_top1"),
                              1.0 / PRIMARY_NWAY),
        "with_mean_fold_mean": summary.get(headline_key.replace("_nomean", ""),
                                           float("nan")),
    }


def _shortcut_block(summary: Mapping[str, float], pk: str) -> Dict[str, Any]:
    """The size-harmonised material control -- and why cross_group is not it."""
    out: Dict[str, Any] = {
        "control": "within-material 2-way top-1 vs overall 2-way top-1",
        "why": ("both galleries hold exactly 2 candidates, so the two numbers are "
                "size-harmonised and directly comparable; the surviving fraction "
                "(within - 0.5) / (overall - 0.5) is the share of pairwise "
                "discriminability that is NOT explained by the 8-way material code"),
        "not_a_control": {
            "keys": "cross_group_*",
            "reason": ("tactus.eval.retrieval.cross_group builds a per-query gallery "
                       "of every other-material item PLUS the true item, so the true "
                       "item is the only one of its material and a pure material "
                       "classifier scores 100%.  It is an upper bound INFLATED by the "
                       "attribute code, not a control for it.  Earlier drafts of this "
                       "module quoted 0.1718 vs chance 0.0683 (+10.4 pts) as if it "
                       "were shortcut-controlled evidence; that reading was wrong.  "
                       "The bug was this module's interpretation, not retrieval.py."),
            "emitted": "suppressed by default; --keep-cross-group restores the keys",
        },
    }
    for scope in ("single", pk):
        for variant in ("video", "video_nomean"):
            ov = summary.get(f"{scope}/{variant}/nway2_top1")
            wi = summary.get(f"{scope}/{variant}/within_group_nway2_top1")
            if ov is None or wi is None:
                continue
            out[f"{scope}/{variant}"] = {
                "overall_nway2_top1": float(ov),
                "within_material_nway2_top1": float(wi),
                "chance": 0.5,
                "surviving_fraction": _surviving_fraction(float(ov), float(wi)),
                "surviving_fraction_note": (
                    "NaN when the overall 2-way effect is < %.3f above chance: the "
                    "ratio's denominator is then noise" % MIN_NWAY2_EFFECT),
                "mean_of_per_fold_ratio": summary.get(
                    f"{scope}/{variant}/material_control_ratio", float("nan")),
                "within_group_size_mean": summary.get(
                    f"{scope}/{variant}/within_group_size_mean", float("nan")),
            }
    return out


def _uncertainty(out_root: Path, rows: Sequence[Mapping[str, Any]],
                 args: argparse.Namespace, headline_key: str) -> Dict[str, Any]:
    """Load the captured rank matrices and run the CI + permutation stage."""
    folds: List[PrimaryFold] = []
    missing: List[int] = []
    reported: Dict[int, float] = {}
    for r in rows:
        i = int(r.get("fold_index", -1))
        if headline_key in r:
            reported[i] = float(r[headline_key])
        path = out_root / "folds" / f"fold{i:03d}_primary.npz"
        if not path.exists():
            missing.append(i)
            continue
        with np.load(path) as d:
            if f"rank_{PRIMARY_VARIANT}" not in d:
                missing.append(i)
                continue
            folds.append(PrimaryFold(
                fold_index=i,
                ranks=d[f"rank_{PRIMARY_VARIANT}"].astype(np.float64),
                true_idx=d["true_idx"].astype(np.int64),
                subject_id=d["subject_id"].astype(np.int64),
                gallery_video_id=d["gallery_video_id"],
                gallery_material=d["gallery_material"],
            ))
    if not folds:
        log.warning("no rank captures on disk (%s); rerun with --force to get a CI "
                    "and a permutation p", ", ".join(f"fold{i:03d}" for i in missing))
        return {}
    if missing:
        log.warning("uncertainty stage is using %d of %d folds; missing captures for %s",
                    len(folds), len(rows), ", ".join(f"fold{i:03d}" for i in missing))
    t0 = time.time()
    ci = subject_bootstrap_ci(folds, n_boot=args.n_boot, seed=args.seed)
    perm = video_permutation_test(folds, n_perm=args.n_perm, seed=args.seed)
    # The CI and the permutation are computed from the captured rank matrices;
    # the headline comes from the shared scorer.  If the two ever disagree the
    # uncertainty statement belongs to a different number than the one quoted,
    # so the agreement is measured and written down rather than assumed.
    diffs = [abs(ci["per_fold"][f.fold_index] - reported[f.fold_index])
             for f in folds if f.fold_index in reported]
    max_diff = float(max(diffs)) if diffs else float("nan")
    if np.isfinite(max_diff) and max_diff > 1e-9:
        log.warning("captured ranks disagree with the reported headline by up to "
                    "%.3g -- the CI and p-value describe a slightly different "
                    "statistic than the quoted one", max_diff)
    log.info("uncertainty stage: %d fold(s), %d subjects, %.1f s",
             len(folds), ci.get("n_subjects", 0), time.time() - t0)
    return {
        "ci": ci,
        "permutation": perm,
        "captures_used": [f.fold_index for f in folds],
        "captures_missing": missing,
        "recomputed_vs_reported_max_abs_diff": max_diff,
    }


def _log_headline(headline: Mapping[str, Any], shortcut: Mapping[str, Any],
                  audit: Mapping[str, Any]) -> None:
    ci = headline.get("ci") or {}
    perm = (headline.get("permutation") or {}).get("video_level") or {}
    log.info(
        "HEADLINE %s = %.4f (fold mean; chance %.4f)%s%s",
        headline.get("key"), headline.get("fold_mean", float("nan")),
        headline.get("chance", float("nan")),
        ("  |  subject bootstrap %.4f [%.4f, %.4f] over n=%d"
         % (ci.get("estimate", float("nan")), ci.get("ci_lo", float("nan")),
            ci.get("ci_hi", float("nan")), ci.get("n_subjects", 0))) if ci else "",
        ("  |  video-level perm p=%.4g (null %.4f +- %.4f, z=%.1f)"
         % (perm.get("p_value", float("nan")), perm.get("null_mean", float("nan")),
            perm.get("null_sd", float("nan")), perm.get("z_score", float("nan"))))
        if perm else "",
    )
    # report the control for the scope the headline lives in, not the first one
    scope = str(headline.get("key", "")).split("/")[0]
    key = f"{scope}/{PRIMARY_VARIANT}"
    if key in shortcut:
        b = shortcut[key]
        log.info("MATERIAL CONTROL (%s): 2-way within-material %.4f vs overall %.4f "
                 "(chance 0.5) -> %.1f%% of pairwise discriminability survives",
                 key, b["within_material_nway2_top1"], b["overall_nway2_top1"],
                 100 * b["surviving_fraction"])
    if audit.get("available"):
        log.info("DECIM AUDIT over %d matched fold(s): pool %.4f vs slice %.4f "
                 "(pool - slice = %+.4f)", audit["n_folds_compared"],
                 audit.get("pool", float("nan")), audit.get("slice", float("nan")),
                 audit.get("delta_pool_minus_slice", float("nan")))
    else:
        log.info("DECIM AUDIT: %s", audit.get("note", "unavailable"))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
