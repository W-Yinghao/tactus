"""TACTUS training loop.

One :class:`Trainer` trains one fold.  It is deliberately dumb about *what* it
optimizes: the encoder, the video projector and the loss are all built from
config through registries, so adding an algorithm means writing one file in
``tactus/losses/`` and changing one key in a YAML.  Nothing in this file needs
to know the loss exists.

What it *is* opinionated about:

**The pseudo-trial curriculum.**  Single-trial EEG at 800 ms SOA has a rotten
SNR, and an InfoNCE-style objective on top of it can spend its first epochs
fitting noise.  The dataset gives us the fix for free: 8 repeats of every one of
360 conditions per subject.  Training starts with ``k=4`` averaged repeats as
the EEG anchor and anneals to ``k=1`` over the first N epochs, so the model is
handed an easy problem first and the *same* problem it will be evaluated on
last.  Evaluation always reports both k=1 and k=4, regardless of the curriculum.

**Batch composition.**  Video embeddings are a frozen 360-entry codebook, so two
trials of the same condition in one batch are an exact false negative pair.  The
default sampler packs each batch with distinct *base videos* (not conditions --
the four orientations of one video are near-duplicates in most video towers),
which is also what makes the in-batch softmax denominator interpretable as a
"n-way" discrimination.

**Validation never touches the test fold.**  Early stopping and checkpoint
selection run on an inner split carved out of the training videos (and, for
subject-generalization regimes, the training subjects), so the reported test
number is not a selection-optimistic one.

**Scaling knobs.**  ``train.n_train_subjects`` subsamples training *subjects*;
``train.train_trial_frac`` / ``train.n_train_trials`` subsample training
*trials*, stratified over the ``(subject, condition)`` design and nested across
fractions.  Both touch training only -- validation and test are the same rows at
every point of a curve, and ``eval_fingerprint.json`` proves it.  A naive trial
subsample moves four things at once; each has an explicit control:

===========================================  ==========================================
confound                                     control
===========================================  ==========================================
(a) fewer trials -> fewer optimizer steps    ``train.steps_per_epoch: full`` re-walks
    and a compressed LR schedule             the smaller pool to the full split's step
                                             count (``null`` = epoch-matched arm)
(b) cells that can no longer supply k        counted and logged (``cells_below_k``,
    repeats for the pseudo-trial curriculum  ``mean_k_avail``) per fraction
(c) EA whitener + robust scaler are fit      ``data.scaling.fit_source:
    on training trials                       subsample|full_train``
(d) ``batch.mode=distinct_video`` needs one  stratification keeps every base video, and
    trial per distinct base video            the realised batch size is logged
===========================================  ==========================================

Everything is resumable (``checkpoints/last.pt`` holds optimizer, scheduler,
scaler and RNG state) and idempotent (a fold with ``done.json`` is skipped).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from tactus.common import (
    EpochStore,
    SubjectTransform,
    VideoEmbeddings,
    apply_continuous_stats,
    atomic_write_json,
    attach_label_ids,
    continuous_stats,
    fit_subject_transforms,
    hash_obj,
    load_label_maps,
    load_trials,
    load_video_emb,
)
from tactus.models.eeg.subject_cond import UNSEEN_SUBJECT as SUBJECT_COND_UNSEEN

log = logging.getLogger(__name__)

__all__ = [
    "TrialView",
    "StratifiedVideoBatchSampler",
    "Trainer",
    "TrainResult",
    "train_fold",
    "set_seed",
    "build_model",
    "stratified_trial_subsample",
    "cell_availability",
    "uid_fingerprint",
]

#: Meta keys handed to every loss (contract D).  Kept in sync with
#: ``tactus.losses.base.META_KEYS``; imported dynamically when available.
ID_META_KEYS = (
    "video_id", "condition_id", "orientation", "subject_id", "sequence_id",
    "material_id", "touch_type_id", "toucher_id", "pain",
)
CONTINUOUS_META_KEYS = ("valence", "arousal", "threat")
META_KEYS = ID_META_KEYS + CONTINUOUS_META_KEYS

#: Sentinel passed for "subject not seen during training"; every real subject
#: uses its raw 1..80 id.  ``SubjectConditioner.rows_for`` maps any negative id
#: to row ``-1``, which triggers the pre-registered unseen rule *computed* from
#: the training subjects (mean-of-train-tokens / identity layer / zero adapter,
#: blueprint 4.2, decision D2) rather than reading a learned row.
#:
#: This must stay negative.  With ``0`` the encoder was handed a real parameter
#: row that no data ever trains, and only ``strict_unseen=True`` masked it -- a
#: silent LOSO-invalidating trap the moment anyone ran a transductive variant.
UNSEEN_SUBJECT_INDEX = SUBJECT_COND_UNSEEN  # == -1


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch and optionally force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def _worker_init(worker_id: int) -> None:  # pragma: no cover - subprocess
    info = torch.utils.data.get_worker_info()
    seed = (int(info.seed) if info is not None else 0) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# trial-count scaling: subsample TRAINING trials only
# --------------------------------------------------------------------------- #

#: Columns that define a stratification cell, per ``train.trial_subsample.stratify``.
_STRATA: Dict[str, Sequence[str]] = {
    "subject_condition": ("subject_id", "condition_id"),
    "subject_video": ("subject_id", "video_id"),
    "subject": ("subject_id",),
    "condition": ("condition_id",),
    "video": ("video_id",),
    "none": (),
}


def uid_fingerprint(uids: Sequence[int]) -> Dict[str, Any]:
    """Byte-level fingerprint of a set of ``trial_uid`` values.

    Two runs whose evaluation set is *literally* the same rows in the same order
    produce the same ``order_sha1``; two runs that merely contain the same trials
    share ``set_sha1``.  Both are written to ``eval_fingerprint.json`` so the
    "the eval set is fixed across the curve" claim is checkable after the fact
    rather than asserted in a docstring.
    """
    a = np.asarray(uids, dtype=np.int64)
    return {
        "n": int(a.size),
        "order_sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "set_sha1": hashlib.sha1(np.ascontiguousarray(np.sort(a)).tobytes()).hexdigest(),
    }


def cell_availability(trials: pd.DataFrame, k: int) -> Dict[str, float]:
    """How well a trial set can still form ``k``-repeat pseudo-trials.

    The curriculum's anchor is the average of ``k`` repeats of one
    ``(subject, condition)`` cell drawn *from the same view*.  ``TrialView``
    degrades gracefully -- it takes ``min(k-1, n_cell-1)`` partners -- so a
    subsampled training set does not crash, it silently trains at a smaller
    effective ``k``.  That is exactly the confound this reports:

    ``cells_below_k``
        cells with fewer than ``k`` surviving trials.
    ``frac_trials_below_k``
        share of anchors that cannot reach ``k``.
    ``mean_k_avail``
        the trial-weighted mean of ``min(n_cell, k)`` -- the *realised* k the
        curriculum will actually train at when it asks for ``k``.
    """
    k = int(k)
    if not len(trials):
        return {"k": float(k), "n_cells": 0.0, "cells_below_k": 0.0,
                "frac_cells_below_k": 0.0, "frac_trials_below_k": 0.0,
                "mean_k_avail": 0.0, "n_full_pseudo_trials": 0.0}
    counts = (
        trials.groupby(["subject_id", "condition_id"], sort=False, observed=True)
        .size().to_numpy(np.int64)
    )
    below = counts < int(k)
    return {
        "k": float(k),
        "n_cells": float(counts.size),
        "cells_below_k": float(below.sum()),
        "frac_cells_below_k": float(below.mean()) if counts.size else 0.0,
        "frac_trials_below_k": float(counts[below].sum() / max(1, counts.sum())),
        "mean_k_avail": float((np.minimum(counts, int(k)) * counts).sum() / max(1, counts.sum())),
        "min_cell": float(counts.min()) if counts.size else 0.0,
        "median_cell": float(np.median(counts)) if counts.size else 0.0,
        "max_cell": float(counts.max()) if counts.size else 0.0,
        "n_full_pseudo_trials": float((counts // int(k)).sum()),
    }


def _joint_code(df: pd.DataFrame, cols: Sequence[str], n: int) -> np.ndarray:
    """Dense 0..m-1 code for the tuple of ``cols`` (all-zeros when ``cols`` empty)."""
    if not cols:
        return np.zeros(n, dtype=np.int64)
    joint = np.zeros(n, dtype=np.int64)
    for c in cols:
        v = df[c].to_numpy(np.int64)
        joint = joint * (int(v.max()) + 2) + v
    return pd.factorize(joint, sort=True)[0].astype(np.int64)


def _nested_key(label: np.ndarray, rng: np.random.Generator) -> tuple:
    """``(q, rank)`` with ``q = (rank_within_label + u_label) / label_size`` in [0, 1).

    Sorting by ``q`` and taking a prefix gives a sample that is balanced across
    labels to within one element, and -- because ``q`` does not depend on how big
    the prefix is -- every smaller prefix is a subset of every larger one.
    ``rank`` is returned rather than recovered from ``q`` so a ``min_per_cell``
    floor is an integer comparison, not a float round-trip.
    """
    n = int(label.size)
    n_lab = int(label.max()) + 1 if n else 0
    size = np.bincount(label, minlength=n_lab).astype(np.int64)
    order = np.lexsort((rng.random(n), label))
    starts = np.concatenate([[0], np.cumsum(size)[:-1]])
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64) - starts[label[order]]
    u = rng.random(n_lab)
    q = (rank.astype(np.float64) + u[label]) / size[label].astype(np.float64)
    return q, rank


def stratified_trial_subsample(
    trials: pd.DataFrame,
    *,
    frac: float | None = None,
    n_trials: int | None = None,
    stratify: str = "subject_condition",
    unit: str = "trial",
    seed: int = 0,
    salt: int = 8821,
    min_per_cell: int = 0,
) -> tuple:
    """Keep a balanced subset of ``trials``; return ``(kept, info)``.

    Why not ``df.sample(frac=f)``: a flat random draw over 114k training trials
    leaves a ragged design -- some ``(subject, condition)`` cells keep 4 repeats
    and some keep 0 -- so a "1/8 of the trials" arm is not 1/8 of the *same*
    experiment, it is a different, partly-missing one.  Every cell here keeps its
    own proportional share, to within one trial.

    The mechanism is a fixed per-trial key ``q = (rank_within_cell + u_cell) / n_cell``
    with ``rank`` a per-cell random permutation and ``u_cell ~ U[0,1)``; the kept
    set is the ``N`` trials with the smallest ``q``.  Three consequences that
    matter for a scaling curve:

    * **nested** -- the 1/8 subset is a subset of the 1/4 subset is a subset of
      1/2, because ``q`` does not depend on the fraction.  The curve therefore
      measures *added data*, not a fresh draw at every point.
    * **balanced** -- within a cell the kept count is ``floor(f*n)`` or
      ``ceil(f*n)``; no cell is starved while another keeps everything.
    * **exact** -- ``len(kept)`` is exactly the requested ``N``.

    ``stratify="none"`` reproduces the flat random draw on purpose, so the cost
    of *not* stratifying can be measured rather than argued about.

    ``unit`` decides *how a smaller trial budget is spent*, and the two answers
    are different experiments:

    ``"trial"`` (default)
        thin every ``(subject, condition)`` cell.  Design coverage is preserved
        -- measured on the real 80-subject within_subject fold, all 224 training
        conditions and all 56 training base videos survive down to 1/8 -- but the
        repeats per cell go with the fraction, so at 1/8 a cell holds ~1 trial
        and the k=4 pseudo-trial curriculum silently collapses to k=1.
    ``"cell"``
        keep *whole* cells (all their repeats) for a proportional subset of
        cells, chosen balanced within each ``stratify`` group.  ``k`` survives
        intact (measured 3.91 at 1/8 on the same fold, against 1.00 for
        ``"trial"``); what shrinks is how many (subject, condition) pairs are
        covered, 17,916 -> 2,223.  The trial count then lands within one cell of
        the target instead of exactly on it.

    Running both at the same fraction is what separates "the model needs more
    repeats of a condition" from "the model needs more conditions".
    """
    n_before = int(len(trials))
    if n_before == 0:
        return trials, {"applied": False, "reason": "empty", "n_before": 0, "n_after": 0}
    if n_trials is not None:
        target = int(n_trials)
    elif frac is not None:
        target = int(round(float(frac) * n_before))
    else:
        return trials, {"applied": False, "reason": "no frac/n_trials requested",
                        "n_before": n_before, "n_after": n_before}
    target = max(1, min(target, n_before))
    if target >= n_before:
        return trials, {"applied": False, "reason": "target >= available",
                        "n_before": n_before, "n_after": n_before, "target": target}

    unit = str(unit)
    if unit not in {"trial", "cell"}:
        raise ValueError(f"train.trial_subsample.unit must be trial|cell, got {unit!r}")
    stratify = str(stratify)
    if unit == "cell" and stratify == "subject_condition":
        # the cell IS (subject, condition); balance the cell draw over subjects
        stratify = "subject"
    cols = _STRATA.get(stratify)
    if cols is None:
        raise ValueError(f"unknown train.trial_subsample.stratify={stratify!r}; "
                         f"choose from {sorted(_STRATA)}")
    missing = [c for c in cols if c not in trials.columns]
    if missing:
        raise KeyError(f"stratify={stratify!r} needs columns {missing}")

    # Sort by trial_uid first so the key assignment -- and therefore the whole
    # subsample -- is invariant to the row order the fold happened to hand us.
    df = trials.sort_values("trial_uid", kind="stable")
    n = len(df)
    rng = np.random.default_rng([int(seed), int(salt)])
    cell = _joint_code(df, cols, n)
    n_cells = int(cell.max()) + 1 if n else 0

    if unit == "cell":
        # whole (subject, condition) cells, balanced within the stratify group
        repeat = _joint_code(df, ("subject_id", "condition_id"), n)
        n_rep = int(repeat.max()) + 1
        rep_size = np.bincount(repeat, minlength=n_rep).astype(np.int64)
        rep_group = np.zeros(n_rep, dtype=np.int64)
        rep_group[repeat] = cell                       # each cell sits in one group
        qrep, _ = _nested_key(rep_group, rng)
        order = np.argsort(qrep, kind="stable")
        cum = np.cumsum(rep_size[order])
        n_keep_cells = int(np.searchsorted(cum, target, side="left") + 1)
        n_keep_cells = min(n_keep_cells, n_rep)
        # take whichever prefix lands closest to the requested trial count
        if n_keep_cells > 1 and abs(int(cum[n_keep_cells - 2]) - target) < abs(
            int(cum[n_keep_cells - 1]) - target
        ):
            n_keep_cells -= 1
        keep_rep = np.zeros(n_rep, dtype=bool)
        keep_rep[order[:n_keep_cells]] = True
        mask = keep_rep[repeat]
    else:
        q, rank = _nested_key(cell, rng)
        if min_per_cell > 0:  # force-keep the first `min_per_cell` of every cell
            q = np.where(rank < int(min_per_cell), q - 10.0, q)
        keep_pos = np.argpartition(q, target - 1)[:target] if target < n else np.arange(n)
        mask = np.zeros(n, dtype=bool)
        mask[keep_pos] = True

    # Report per-cell balance over the (subject, condition) repeat cell whatever
    # `stratify` was, so the number means the same thing in every arm and lines
    # up with cell_availability() and the k diagnostics.
    repeat_cell = _joint_code(df, ("subject_id", "condition_id"), n)
    kept_sizes = np.bincount(
        repeat_cell[mask], minlength=int(repeat_cell.max()) + 1
    ).astype(np.int64)
    # sort_index restores the caller's row order (df was re-ordered by trial_uid)
    kept = df.iloc[np.flatnonzero(mask)].sort_index(kind="stable").reset_index(drop=True)

    info: Dict[str, Any] = {
        "applied": True,
        "stratify": str(stratify),
        "unit": unit,
        "seed": int(seed), "salt": int(salt), "min_per_cell": int(min_per_cell),
        "requested_frac": None if frac is None else float(frac),
        "requested_n_trials": None if n_trials is None else int(n_trials),
        "n_before": n_before,
        "n_after": int(len(kept)),
        "target": int(target),
        "realised_frac": float(len(kept) / n_before),
        "n_strata": int(n_cells),
        # per (subject, condition) repeat cell, whatever `stratify` was
        "n_cells": int(kept_sizes.size),
        "cells_emptied": int((kept_sizes == 0).sum()),
        "kept_per_cell_min": int(kept_sizes.min()) if kept_sizes.size else 0,
        "kept_per_cell_median": float(np.median(kept_sizes)) if kept_sizes.size else 0.0,
        "kept_per_cell_max": int(kept_sizes.max()) if kept_sizes.size else 0,
        "n_subjects_before": int(trials["subject_id"].nunique()),
        "n_subjects_after": int(kept["subject_id"].nunique()),
        "n_videos_before": int(trials["video_id"].nunique()),
        "n_videos_after": int(kept["video_id"].nunique()),
        "n_conditions_before": int(trials["condition_id"].nunique()),
        "n_conditions_after": int(kept["condition_id"].nunique()),
    }
    return kept, info


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


class TrialView(Dataset):
    """A view over a subset of the trial table, with the curriculum built in.

    One item = one *anchor*.  With ``k == 1`` the anchor is a single trial; with
    ``k > 1`` it is the average of that trial and ``k-1`` other repeats of the
    same ``(subject, condition)`` **drawn from this view only** (so a validation
    or test pseudo-trial can never borrow a training trial, and vice versa).

    The dataset length is independent of ``k``: every trial seeds exactly one
    anchor per epoch.  That keeps the number of optimizer steps, the LR schedule
    and the wall-clock cost of an epoch constant while ``k`` anneals.
    """

    def __init__(
        self,
        trials: pd.DataFrame,
        store: EpochStore,
        video: VideoEmbeddings,
        transforms: Mapping[int, SubjectTransform],
        *,
        subject_index: Mapping[int, int] | None = None,
        k: int = 1,
        p_single: float = 0.0,
        rescale: str = "none",
        seed: int = 0,
    ) -> None:
        required = {"subject_id", "within_subj_idx", "video_id", "condition_id", "trial_uid"}
        missing = required - set(trials.columns)
        if missing:
            raise KeyError(f"trial table is missing required columns: {sorted(missing)}")
        self.trials = trials.reset_index(drop=True)
        self.store = store
        self.video = video
        self.transforms = dict(transforms)
        self.subject_index = dict(subject_index or {})
        self.rescale = str(rescale)
        self.seed = int(seed)
        self._k = int(k)
        self._p_single = float(p_single)
        self._epoch = 0

        # column caches: __getitem__ must not touch pandas (slow + not fork-safe)
        t = self.trials
        self.subject = t["subject_id"].to_numpy(np.int64)
        self.within = t["within_subj_idx"].to_numpy(np.int64)
        self.video_id = t["video_id"].to_numpy(np.int64)
        self.condition_id = t["condition_id"].to_numpy(np.int64)
        self.trial_uid = t["trial_uid"].to_numpy(np.int64)
        self.sequence_id = (
            t["sequence_id"].to_numpy(np.int64) if "sequence_id" in t else np.zeros(len(t), np.int64)
        )
        self.subject_idx = np.array(
            [self.subject_index.get(int(s), UNSEEN_SUBJECT_INDEX) for s in self.subject],
            dtype=np.int64,
        )
        for key in ("material_id", "touch_type_id", "toucher_id"):
            setattr(self, key, t[key].to_numpy(np.int64) if key in t else np.full(len(t), -1, np.int64))
        self.pain = t["pain"].to_numpy(np.int64) if "pain" in t else np.zeros(len(t), np.int64)
        for key in CONTINUOUS_META_KEYS:
            col = f"{key}_z" if f"{key}_z" in t else key
            setattr(
                self, f"c_{key}",
                t[col].to_numpy(np.float32) if col in t else np.zeros(len(t), np.float32),
            )

        # (subject, condition) -> row positions, for pseudo-trial partners
        key = self.subject.astype(np.int64) * 1000 + self.condition_id
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        bounds = np.flatnonzero(np.diff(sorted_key)) + 1
        self._group_rows = np.split(order, bounds)
        self._group_of = np.empty(len(t), dtype=np.int64)
        for gi, rows in enumerate(self._group_rows):
            self._group_of[rows] = gi
        self.n_times = store.n_times

    # -- curriculum knobs --------------------------------------------------- #

    def set_epoch(self, epoch: int, k: int, p_single: float) -> None:
        """Set the curriculum state for the coming epoch (call before the loader)."""
        self._epoch = int(epoch)
        self._k = max(1, int(k))
        self._p_single = float(np.clip(p_single, 0.0, 1.0))

    @property
    def k(self) -> int:
        return self._k

    def __len__(self) -> int:
        return len(self.trials)

    # -- loading ------------------------------------------------------------ #

    def load_x(self, rows: np.ndarray) -> np.ndarray:
        """Average the epochs at dataset rows ``rows`` and normalize.

        All rows must belong to the same subject (they always do -- partners come
        from a ``(subject, condition)`` group).
        """
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        sid = int(self.subject[rows[0]])
        x = self.store.take(sid, self.within[rows])
        x = x.mean(axis=0) if x.shape[0] > 1 else x[0]
        if self.rescale == "sqrt_k" and rows.size > 1:
            x = x * math.sqrt(rows.size)
        tf = self.transforms.get(sid)
        if tf is not None:
            x = tf.apply(x)
        return np.ascontiguousarray(x, dtype=np.float32)

    def _partners(self, i: int, k: int, rng: np.random.Generator) -> np.ndarray:
        if k <= 1:
            return np.array([i], dtype=np.int64)
        pool = self._group_rows[self._group_of[i]]
        if pool.size <= 1:
            return np.array([i], dtype=np.int64)
        others = pool[pool != i]
        take = min(k - 1, others.size)
        picked = rng.choice(others, size=take, replace=False)
        return np.concatenate([[i], picked]).astype(np.int64)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        i = int(i)
        rng = np.random.default_rng([self.seed, self._epoch, i])
        k = 1 if (self._p_single > 0 and rng.random() < self._p_single) else self._k
        rows = self._partners(i, k, rng)
        x = self.load_x(rows)
        v = self.video.cond_emb[self.condition_id[i]]
        item: Dict[str, torch.Tensor] = {
            "x": torch.from_numpy(x),
            "v": torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)),
            "row": torch.tensor(i, dtype=torch.long),
            "trial_uid": torch.tensor(self.trial_uid[i], dtype=torch.long),
            "k": torch.tensor(rows.size, dtype=torch.long),
            "subject_index": torch.tensor(self.subject_idx[i], dtype=torch.long),
            "video_id": torch.tensor(self.video_id[i], dtype=torch.long),
            "condition_id": torch.tensor(self.condition_id[i], dtype=torch.long),
            "orientation": torch.tensor(self.condition_id[i] % 4, dtype=torch.long),
            "subject_id": torch.tensor(self.subject[i], dtype=torch.long),
            "sequence_id": torch.tensor(self.sequence_id[i], dtype=torch.long),
            "material_id": torch.tensor(self.material_id[i], dtype=torch.long),
            "touch_type_id": torch.tensor(self.touch_type_id[i], dtype=torch.long),
            "toucher_id": torch.tensor(self.toucher_id[i], dtype=torch.long),
            "pain": torch.tensor(self.pain[i], dtype=torch.long),
        }
        for key in CONTINUOUS_META_KEYS:
            item[key] = torch.tensor(getattr(self, f"c_{key}")[i], dtype=torch.float32)
        return item


class PseudoTrialView(Dataset):
    """Disjoint k-repeat pseudo-trials over a :class:`TrialView` (evaluation only).

    Each ``(subject, condition)`` group of ``n`` trials yields ``floor(n / k)``
    *disjoint* pseudo-trials, so no trial contributes to two of them.  Resampling
    (a different shuffle per ``resample`` index) is what turns this into the
    "k=4 repeats averaged, resampled" estimator the eval contract asks for.
    """

    def __init__(self, base: TrialView, k: int = 4, resample: int = 0, seed: int = 0) -> None:
        self.base = base
        self.k = max(1, int(k))
        rng = np.random.default_rng([seed, resample, self.k])
        groups: List[np.ndarray] = []
        for rows in base._group_rows:
            if rows.size < self.k:
                continue
            perm = rng.permutation(rows)
            n_full = perm.size // self.k
            groups.extend(np.split(perm[: n_full * self.k], n_full))
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rows = self.groups[i]
        head = int(rows[0])
        item = self.base[head]  # meta of the first member (all share subject+condition)
        item["x"] = torch.from_numpy(self.base.load_x(rows))
        item["k"] = torch.tensor(len(rows), dtype=torch.long)
        return item


def collate(batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of items into a batch dict (all values are tensors)."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# --------------------------------------------------------------------------- #
# batch sampler
# --------------------------------------------------------------------------- #


class StratifiedVideoBatchSampler(Sampler):
    """Pack batches with distinct base videos (blueprint 4.3).

    Two modes:

    ``distinct_video``
        ``B`` different base videos, one trial each.  The in-batch softmax is
        then a clean B-way discrimination with no false negatives.
    ``video_x_subject``
        ``B / S`` different base videos x ``S`` trials each, the ``S`` drawn from
        *different subjects*.  This is the composition a cross-subject (CLISA)
        term needs -- it has same-condition positives across subjects in the
        batch -- and it deliberately reintroduces duplicate videos, which the
        loss is responsible for masking.

    Every epoch consumes every trial at most once; batches are formed from the
    videos with the most trials still unused, which keeps the composition
    balanced without an explicit quota.

    ``n_batches`` breaks the "at most once" rule *on purpose*: it pins the number
    of optimizer steps in an epoch, re-shuffling and re-walking the trial pool as
    many times as it takes.  That is what makes a trial-count scaling curve
    interpretable -- without it, "1/8 of the trials" also means "1/8 of the
    gradient steps, and a cosine schedule squeezed into 1/8 of the updates", and
    a drop in accuracy cannot be attributed to the data.

    Known limitation of that mode: ``TrialView.__getitem__`` seeds its
    pseudo-trial draw on ``(seed, epoch, row)``, so the *second* visit to a row
    inside one epoch rebuilds the *same* anchor from the same partners.  Batch
    composition still differs between passes; the anchor does not.  Across
    epochs everything redraws.  It is a small loss of stochasticity in the
    step-matched arm, not a leak, but it is why the epoch-matched arm is worth
    running alongside rather than instead.
    """

    def __init__(
        self,
        video_id: np.ndarray,
        subject_id: np.ndarray,
        *,
        batch_size: int,
        mode: str = "distinct_video",
        subjects_per_video: int = 1,
        drop_last: bool = True,
        seed: int = 0,
        max_batches: int | None = None,
        n_batches: int | None = None,
    ) -> None:
        self.video_id = np.asarray(video_id, dtype=np.int64)
        self.subject_id = np.asarray(subject_id, dtype=np.int64)
        self.mode = str(mode)
        self.subjects_per_video = max(1, int(subjects_per_video))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.max_batches = max_batches
        self.n_batches = None if n_batches is None else max(1, int(n_batches))
        self._epoch = 0
        self.passes_per_epoch = 1.0

        uniq = np.unique(self.video_id)
        self.n_videos = int(uniq.size)
        if self.mode == "video_x_subject":
            n_groups = max(1, int(batch_size) // self.subjects_per_video)
            self.n_video_slots = min(n_groups, self.n_videos)
            self.batch_size = self.n_video_slots * self.subjects_per_video
        else:
            self.batch_size = min(int(batch_size), self.n_videos)
            self.n_video_slots = self.batch_size
        if self.batch_size < int(batch_size):
            log.warning(
                "batch size clamped %d -> %d: only %d distinct base videos in this "
                "split (mode=%s).", int(batch_size), self.batch_size, self.n_videos, self.mode,
            )
        self._buckets: Dict[int, np.ndarray] = {
            int(v): np.flatnonzero(self.video_id == v) for v in uniq
        }
        self._one_pass_batches = 0
        self._len = sum(1 for _ in self._generate(0))
        self.batches_per_pass = self._one_pass_batches
        # how many times an epoch walks the (subsampled) trial pool
        self.passes_per_epoch = (
            float(self._len) / max(1, self._one_pass_batches) if self._one_pass_batches else 0.0
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._generate(self._epoch)

    def _cap(self) -> int | None:
        caps = [c for c in (self.max_batches, self.n_batches) if c is not None]
        return min(int(c) for c in caps) if caps else None

    def _generate(self, epoch: int) -> Iterator[List[int]]:
        rng = np.random.default_rng([self.seed, epoch])
        cap = self._cap()
        n_emitted = 0
        one_pass = 0
        for cycle in itertools.count():
            pool = {v: list(rng.permutation(rows)) for v, rows in self._buckets.items()}
            emitted_this_pass = 0
            for batch in self._walk_pool(pool, rng):
                yield batch
                n_emitted += 1
                emitted_this_pass += 1
                if cap is not None and n_emitted >= cap:
                    if cycle == 0:
                        one_pass = emitted_this_pass
                    self._one_pass_batches = one_pass or emitted_this_pass
                    return
            if cycle == 0:
                one_pass = emitted_this_pass
            # a single pass over the pool is one "epoch" unless a step target
            # asks for more; stop if the pool cannot yield a batch at all, or
            # this would loop forever on a split too small to fill one batch
            if self.n_batches is None or emitted_this_pass == 0:
                break
        self._one_pass_batches = one_pass

    def _walk_pool(
        self, pool: Dict[int, List[int]], rng: np.random.Generator
    ) -> Iterator[List[int]]:
        """Emit batches until ``pool`` (one full pass over the trials) is spent."""
        while True:
            live = [(len(r), v) for v, r in pool.items() if r]
            if not live:
                return
            need = self.n_video_slots
            if len(live) < need:
                if self.drop_last:
                    return
                need = len(live)
            # most-remaining first, ties broken randomly, so no video starves
            counts = np.array([c for c, _ in live], dtype=np.int64)
            vids = np.array([v for _, v in live], dtype=np.int64)
            order = np.lexsort((rng.random(len(live)), -counts))
            chosen = vids[order[:need]]

            batch: List[int] = []
            for v in chosen:
                rows = pool[int(v)]
                if self.mode == "video_x_subject":
                    picked = self._pick_distinct_subjects(rows, self.subjects_per_video)
                else:
                    picked = [rows.pop()]
                batch.extend(int(r) for r in picked)
            if self.drop_last and len(batch) < self.batch_size:
                return
            if not batch:
                return
            yield batch

    def _pick_distinct_subjects(self, rows: List[int], s: int) -> List[int]:
        """Pop up to ``s`` rows from ``rows``, preferring distinct subjects."""
        picked: List[int] = []
        seen: set = set()
        skipped: List[int] = []
        while rows and len(picked) < s:
            r = rows.pop()
            sid = int(self.subject_id[r])
            if sid in seen:
                skipped.append(r)
                if len(skipped) > 4 * s:  # give up on distinctness, take it
                    picked.append(r)
                continue
            seen.add(sid)
            picked.append(r)
        rows.extend(skipped)
        return picked


# --------------------------------------------------------------------------- #
# model construction (registry-agnostic)
# --------------------------------------------------------------------------- #


def _resolve(candidates: Sequence[tuple], what: str) -> Callable:
    """Return the first importable ``module.attr`` from ``candidates``."""
    tried = []
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # module not written yet / import error inside
            tried.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue
        obj = getattr(mod, attr, None)
        if obj is not None:
            return obj
        tried.append(f"{mod_name}.{attr}: absent")
    raise ImportError(
        f"could not resolve {what}. Tried:\n  " + "\n  ".join(tried)
    )


def _flex_call(fn: Callable, name: str | None, params: Mapping[str, Any]) -> Any:
    """Call a builder/class with whatever calling convention it exposes.

    Tries, in order: ``fn(cfg_mapping)``, ``fn(name=..., **params)``,
    ``fn(**params_filtered_to_signature)``.  Model builders across the codebase
    are written by different hands; this keeps a signature mismatch from being a
    40-fold-sweep-killing crash on hour three.
    """
    params = {k: v for k, v in dict(params).items()}
    errors: List[str] = []
    try:
        sig = inspect.signature(fn)
        pnames = list(sig.parameters)
    except (TypeError, ValueError):  # builtins / C-extensions
        sig, pnames = None, []

    if pnames and pnames[0] in {"cfg", "config", "spec", "opts"}:
        cfg = dict(params)
        if name is not None:
            cfg.setdefault("name", name)
        try:
            return fn(cfg)
        except TypeError as exc:
            errors.append(f"fn(cfg): {exc}")
    if "name" in pnames and name is not None:
        try:
            return fn(name=name, **params)
        except TypeError as exc:
            errors.append(f"fn(name=..., **params): {exc}")
    if sig is not None and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        filtered = {k: v for k, v in params.items() if k in pnames}
    else:
        filtered = params
    try:
        return fn(**filtered)
    except TypeError as exc:
        errors.append(f"fn(**params): {exc}")
    raise TypeError(
        f"could not call {getattr(fn, '__name__', fn)!r} with name={name!r} and "
        f"params={sorted(params)}. Attempts:\n  " + "\n  ".join(errors)
    )


def build_model(cfg: Mapping[str, Any], *, n_times: int, d_video: int) -> tuple:
    """Build ``(eeg_encoder, video_projector)`` from ``cfg.model``.

    The EEG encoder must satisfy contract F: ``forward(x, subject_id) -> (B, D)``
    L2-normalized, with ``subject_id`` an index into a subject table whose row
    ``0`` is the mean/unseen-subject token.
    """
    mcfg = dict(cfg)
    d_embed = int(mcfg.get("d_embed", 256))
    eeg_cfg = dict(mcfg.get("eeg", {}) or {})
    eeg_name = eeg_cfg.pop("name", "nice")
    eeg_params = dict(eeg_cfg.pop("params", {}) or {})
    eeg_params.setdefault("n_channels", int(mcfg.get("n_channels", 64)))
    eeg_params.setdefault("n_times", int(n_times))
    eeg_params.setdefault("d_embed", d_embed)
    eeg_params.setdefault("d_out", d_embed)
    # D2: NO "+1" row.  SubjectConditioner already maps parameter rows to dataset
    # ids 1..n_subjects, so an extra row 0 would be a never-trained embedding that
    # a held-out subject could be routed into.  Unseen subjects get index -1 and
    # the pre-registered computed rule (mean-of-train-tokens / identity / zero).
    eeg_params.setdefault("n_subjects", int(mcfg.get("n_subjects", 80)))
    eeg_params.setdefault("subject_conditioning", mcfg.get("subject_conditioning", "token"))
    eeg_params.update(eeg_cfg)

    builder = _resolve(
        [
            ("tactus.models.eeg", "build_eeg_encoder"),
            ("tactus.models.eeg.base", "build_eeg_encoder"),
            ("tactus.models.eeg", "build_encoder"),
            ("tactus.models.eeg.base", "build_encoder"),
            ("tactus.models", "build_eeg_encoder"),
            ("tactus.models.eeg", "get_encoder"),
            ("tactus.models.eeg.base", "get_encoder"),
        ],
        "an EEG encoder builder (expected tactus.models.eeg.base.build_eeg_encoder)",
    )
    encoder = _flex_call(builder, eeg_name, eeg_params)
    if inspect.isclass(encoder):  # a registry returned the class, not an instance
        encoder = _flex_call(encoder, None, eeg_params)

    pcfg = dict(mcfg.get("projector", {}) or {})
    p_name = pcfg.pop("name", "mlp")
    p_params = dict(pcfg.pop("params", {}) or {})
    # canonical names; build_video_projector aliases d_in/d_video -> in_dim and
    # d_out/d_embed -> out_dim, but passing the canonical key removes any doubt
    # about which of two aliases wins
    p_params.setdefault("in_dim", int(d_video))
    p_params.setdefault("out_dim", d_embed)
    p_params.update(pcfg)
    proj_obj = _resolve(
        [
            ("tactus.models.heads", "build_video_projector"),
            ("tactus.models.heads", "VideoProjector"),
            ("tactus.models", "VideoProjector"),
        ],
        "a video projector (expected tactus.models.heads.VideoProjector)",
    )
    projector = _flex_call(proj_obj, p_name, p_params)
    if inspect.isclass(projector):
        projector = _flex_call(projector, None, p_params)
    return encoder, projector


def build_loss_from_cfg(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the loss from ``cfg.loss`` via the loss registry.

    ``tactus.losses`` is imported first so that every ``@register_loss``
    decorator has run before the lookup.
    """
    with contextlib.suppress(Exception):
        importlib.import_module("tactus.losses")
    build = _resolve(
        [("tactus.losses.base", "build_loss"), ("tactus.losses", "build_loss")],
        "build_loss (expected tactus.losses.base.build_loss)",
    )
    return build(dict(cfg))


# --------------------------------------------------------------------------- #
# retrieval helpers
# --------------------------------------------------------------------------- #


def _retrieval_metrics_fn() -> Callable:
    """The project's ``retrieval_metrics`` if it exists, else a local twin."""
    try:
        mod = importlib.import_module("tactus.eval.retrieval")
        fn = getattr(mod, "retrieval_metrics", None)
        if fn is not None:
            return fn
    except Exception:
        pass
    log.debug("tactus.eval.retrieval unavailable; using the in-trainer fallback.")
    return _fallback_retrieval_metrics


def _fallback_retrieval_metrics(
    z_eeg: np.ndarray,
    z_vid_gallery: np.ndarray,
    true_idx: np.ndarray,
    groups: np.ndarray | None = None,
) -> Dict[str, float]:
    """top1 / top5 / mean_rank (+ within-group top1) over a full gallery.

    Ranks are computed with strict ``>`` so exact ties favour the true item;
    cosine ties are measure-zero in float32 but a degenerate (collapsed) encoder
    would otherwise report rank 1 for everything, which is worth knowing is
    *not* what this does.
    """
    z_eeg = np.asarray(z_eeg, dtype=np.float32)
    gal = np.asarray(z_vid_gallery, dtype=np.float32)
    true_idx = np.asarray(true_idx, dtype=np.int64)
    sim = z_eeg @ gal.T
    true_score = sim[np.arange(sim.shape[0]), true_idx]
    rank = 1 + (sim > true_score[:, None]).sum(axis=1)
    out = {
        "top1": float((rank == 1).mean()),
        "top5": float((rank <= 5).mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(np.median(rank)),
        "n": int(sim.shape[0]),
        "gallery": int(gal.shape[0]),
    }
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] == gal.shape[0]:
            g_true = groups[true_idx]
            same = groups[None, :] == g_true[:, None]
            masked = np.where(same, sim, -np.inf)
            r = 1 + (masked > true_score[:, None]).sum(axis=1)
            out["within_group_top1"] = float((r == 1).mean())
            out["within_group_size"] = float(same.sum(axis=1).mean())
    return out


def _subsampled_metrics(
    sim: np.ndarray, true_idx: np.ndarray, size: int, n_draws: int, rng: np.random.Generator
) -> Dict[str, float]:
    """Retrieval into a random gallery of ``size`` that always contains the truth.

    Distractors are drawn **without replacement** (see
    ``tactus.baselines.linear_align.draw_distractors``): with replacement, an
    18-way gallery drawn from 72 conditions loses ~2 distractors to collisions
    and the headline 18-way top-1 is biased upward.
    """
    n, g = sim.shape
    true_score = sim[np.arange(n), true_idx]
    if size >= g:
        rank = 1 + (sim > true_score[:, None]).sum(axis=1)
        return {"top1": float((rank == 1).mean()), "top5": float((rank <= 5).mean()),
                "mean_rank": float(rank.mean())}
    t1, t5, mr = [], [], []
    for _ in range(int(n_draws)):
        keys = rng.random((n, g - 1), dtype=np.float32)
        cand = np.argpartition(keys, size - 2, axis=1)[:, : size - 1]
        cand = cand + (cand >= true_idx[:, None])  # skip the true column
        s = np.take_along_axis(sim, cand, axis=1)
        rank = 1 + (s > true_score[:, None]).sum(axis=1)
        t1.append(float((rank == 1).mean()))
        t5.append(float((rank <= 5).mean()))
        mr.append(float(rank.mean()))
    return {"top1": float(np.mean(t1)), "top1_sd": float(np.std(t1)),
            "top5": float(np.mean(t5)), "mean_rank": float(np.mean(mr))}


# --------------------------------------------------------------------------- #
# config plumbing
# --------------------------------------------------------------------------- #


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    """``_get(cfg, "train.optimizer.lr", 3e-4)`` for dicts and OmegaConf alike."""
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if hasattr(cur, "get") and not isinstance(cur, (list, tuple)):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def _to_container(cfg: Any) -> Dict[str, Any]:
    with contextlib.suppress(Exception):
        from omegaconf import OmegaConf  # type: ignore

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, Mapping):
        return {k: _to_container(v) if isinstance(v, Mapping) else v for k, v in cfg.items()}
    return cfg


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


@dataclass
class TrainResult:
    """Everything run.py needs to know about one finished fold."""

    run_dir: str
    fold_key: str
    regime: str
    status: str = "ok"
    best_epoch: int = -1
    best_monitor: float = float("nan")
    epochs_run: int = 0
    seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_train_subjects: int = 0
    n_test_videos: int = 0
    message: str = ""
    # --- trial-count scaling curve (all flat, so aggregate() picks them up as
    # columns of summary_folds.csv next to the metrics they explain) ---------- #
    n_train_full: int = 0            # training trials before the subsample
    train_trial_frac: float = 1.0    # realised, not requested
    steps_per_epoch: int = 0
    optimizer_steps: int = 0         # steps_per_epoch * epochs_run / accum
    step_matching: str = "epoch_matched"
    passes_per_epoch: float = 1.0
    train_batch_size: int = 0
    n_train_videos: int = 0
    cells_below_k: int = 0           # (subject, condition) cells with < k_start trials
    frac_anchors_below_k: float = 0.0
    mean_k_avail: float = 0.0
    transform_fit_median: float = 0.0   # trials/subject used to fit EA + scaler
    test_uid_sha1: str = ""             # identical across the whole curve, or the
    val_uid_sha1: str = ""              # curve is measuring two different things

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# the trainer
# --------------------------------------------------------------------------- #


class Trainer:
    """Trains one fold and evaluates it.

    Parameters
    ----------
    cfg
        Fully resolved config (plain dict or OmegaConf).
    fold
        A ``tactus.data.splits.Fold``; only ``train_uid``, ``test_uid``,
        ``train_subjects``, ``test_videos``, ``regime`` and the fold ids are used,
        so a duck-typed stand-in works.
    trials
        The full trial table (already post-target-filtered).
    run_dir
        Where checkpoints, logs and metrics go.
    """

    def __init__(
        self,
        cfg: Any,
        fold: Any,
        trials: pd.DataFrame,
        run_dir: Path | str,
        *,
        device: str | torch.device | None = None,
        store: EpochStore | None = None,
        video: VideoEmbeddings | None = None,
    ) -> None:
        self.cfg = _to_container(cfg)
        self.fold = fold
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        self.seed = int(_get(self.cfg, "run.seed", 0))
        self.deterministic = bool(_get(self.cfg, "train.deterministic", False))
        set_seed(self.seed, self.deterministic)

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.window = str(_get(self.cfg, "data.window", "w0600"))
        self.store = store if store is not None else EpochStore(self.window)
        tag = str(_get(self.cfg, "video.model_tag", "siglip2_base_mean"))
        self.video = video if video is not None else load_video_emb(tag)

        self.amp_mode = str(_get(self.cfg, "train.amp", "bf16")).lower()
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.amp_mode)
        self.use_amp = self.amp_dtype is not None and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.use_amp and self.amp_dtype is torch.float16)
        )

        self._prepare_data(trials)
        self._build()
        self._write_budget_record()   # now with the realised step count

        self.epoch = 0
        self.global_step = 0
        self.best_monitor = -math.inf
        self.best_epoch = -1
        self.epochs_since_best = 0
        self.history: List[Dict[str, Any]] = []
        self._log_path = self.run_dir / "log.jsonl"
        self.retrieval_metrics = _retrieval_metrics_fn()

    # -- data --------------------------------------------------------------- #

    def _prepare_data(self, trials: pd.DataFrame) -> None:
        cfg = self.cfg
        maps = load_label_maps(trials)
        trials = attach_label_ids(trials, maps)

        train_uid = np.asarray(getattr(self.fold, "train_uid"), dtype=np.int64)
        test_uid = np.asarray(getattr(self.fold, "test_uid"), dtype=np.int64)
        by_uid = trials.set_index("trial_uid", drop=False)
        tr = by_uid.reindex(train_uid).dropna(subset=["subject_id"]).reset_index(drop=True)
        te = by_uid.reindex(test_uid).dropna(subset=["subject_id"]).reset_index(drop=True)
        for df in (tr, te):
            for c in ("subject_id", "within_subj_idx", "video_id", "condition_id",
                      "trial_uid", "sequence_id", "presentation_number", "pain",
                      "material_id", "touch_type_id", "toucher_id"):
                if c in df.columns:
                    df[c] = df[c].astype(np.int64)

        # subject subsampling for the scaling curve (10/20/40/79 training subjects)
        n_sub = _get(cfg, "train.n_train_subjects", None)
        if n_sub:
            rng = np.random.default_rng([self.seed, 7717, int(n_sub)])
            avail = np.sort(tr["subject_id"].unique())
            keep = np.sort(rng.choice(avail, size=min(int(n_sub), avail.size), replace=False))
            tr = tr.loc[tr["subject_id"].isin(keep)].reset_index(drop=True)
            log.info("scaling curve: training on %d/%d subjects", keep.size, avail.size)

        # inner validation split (never the test fold)
        val_mode = str(_get(cfg, "train.val.mode", "video"))
        tr, va = _inner_val_split(
            tr,
            mode=val_mode,
            video_frac=float(_get(cfg, "train.val.video_frac", 0.2)),
            subject_frac=float(_get(cfg, "train.val.subject_frac", 0.15)),
            seed=self.seed,
        )

        # ---- trial-count scaling curve ------------------------------------- #
        # Subsample the TRAINING trials only.  The split happened first, so the
        # inner validation set (checkpoint selection) and the test fold (the
        # reported number) are literally the same rows at every point on the
        # curve; the fingerprints below make that checkable.
        tr_full = tr
        tr, self.subsample_info = self._subsample_train(tr)
        self.trial_subsample_applied = bool(self.subsample_info.get("applied", False))
        if self.trial_subsample_applied and bool(
            _get(cfg, "train.trial_subsample.include_val", False)
        ):
            va, val_info = self._subsample_train(va)
            self.subsample_info["val"] = val_info
            log.warning(
                "train.trial_subsample.include_val=true: the inner validation set "
                "shrank %d -> %d, so checkpoint selection is NOT identical across "
                "the curve and val/* metrics are not comparable between fractions.",
                val_info.get("n_before"), val_info.get("n_after"),
            )

        # continuous attribute z-scoring: statistics from the FULL training split.
        # These are stimulus-level attributes and every video survives a
        # stratified subsample, so recomputing them per fraction would only add
        # a nuisance difference between arms of the curve.
        self.cont_stats = continuous_stats(tr_full)
        tr, va, te = (apply_continuous_stats(d, self.cont_stats) for d in (tr, va, te))
        tr_full = apply_continuous_stats(tr_full, self.cont_stats)

        train_subjects = np.sort(tr["subject_id"].unique())
        # Identity map over TRAIN subjects only.  Anything absent (i.e. a held-out
        # subject) is mapped to UNSEEN_SUBJECT_INDEX = -1 by TrialView, which is the
        # pre-registered unseen rule (D2) rather than a learned row.
        self.subject_index = {int(s): int(s) for s in train_subjects}
        lost = sorted(set(tr_full["subject_id"].unique()) - set(train_subjects))
        if lost:
            log.warning(
                "trial subsampling removed every training trial of %d subject(s) "
                "%s -- they are now UNSEEN at evaluation, which changes the regime, "
                "not just the trial budget.", len(lost), lost[:10],
            )

        # per-subject label-free signal transforms
        ea_cfg = _get(cfg, "data.ea", {}) or {}
        fit_kwargs = dict(
            ea=bool(ea_cfg.get("enabled", True)),
            ea_shrinkage=float(ea_cfg.get("shrinkage", 0.05)),
            robust_scale=str(_get(cfg, "data.scaling.mode", "robust")) == "robust",
            clamp=float(_get(cfg, "data.scaling.clamp", 20.0)),
            max_trials=_get(cfg, "data.scaling.fit_trials", 512),
            seed=self.seed,
        )
        # (c) of the trial-scaling confounds: the EA whitener and the robust
        # scaler are fit from training trials, so a subsample degrades them too.
        # ``data.scaling.fit_source`` says which arm this run is:
        #   "subsample"  (default) -- honest: the transform sees only the trials
        #                 the run is allowed to have.  Curve = data effect *and*
        #                 preprocessing effect.
        #   "full_train" -- the transform is fit from the full training split, so
        #                 the curve isolates the effect on the *objective*.
        # Both are legitimate; running both is how the confound gets quantified.
        fit_source = str(_get(cfg, "data.scaling.fit_source", "subsample")).lower()
        if fit_source not in {"subsample", "full_train"}:
            raise ValueError(f"data.scaling.fit_source must be subsample|full_train, "
                             f"got {fit_source!r}")
        fit_df = pd.concat([tr_full if fit_source == "full_train" else tr, va])
        # separate cache files: one shared file would be overwritten by whichever
        # call ran last, and the next resume would silently lose the other half.
        # The subsample signature is in the name so a re-run at a different
        # fraction inside the same directory cannot silently reuse the transform
        # fit on a different amount of data.
        sig = ""
        if self.trial_subsample_applied and fit_source == "subsample":
            sig = "_" + hash_obj({k: self.subsample_info.get(k) for k in
                                  ("n_after", "stratify", "seed", "salt")}, 8)
        tf_train = fit_subject_transforms(
            fit_df, self.store,
            cache_path=self.run_dir / f"subject_transforms_train{sig}.json", **fit_kwargs,
        )
        n_fit = np.array([int(getattr(t, "n_fit", 0)) for t in tf_train.values()] or [0])
        avail = fit_df.groupby("subject_id", observed=True).size().to_numpy(np.int64)
        self.transform_fit_stats = {
            "fit_source": fit_source,
            # the inner validation split is NEVER subsampled and it feeds this
            # fit, so confound (c) has a floor: a subject's whitener never sees
            # fewer trials than its val rows, whatever the fraction.
            "includes_val": True,
            "n_available_median": float(np.median(avail)) if avail.size else 0.0,
            "n_available_min": int(avail.min()) if avail.size else 0,
            "fit_trials_cap": None if fit_kwargs["max_trials"] is None else int(fit_kwargs["max_trials"]),
            "n_subjects": int(n_fit.size),
            "n_fit_min": int(n_fit.min()), "n_fit_median": float(np.median(n_fit)),
            "n_fit_max": int(n_fit.max()),
            "n_at_cap": int((n_fit >= int(fit_kwargs["max_trials"] or 0)).sum())
            if fit_kwargs["max_trials"] else 0,
        }
        log.info("subject transforms (%s): trials/subject min=%d median=%.0f max=%d "
                 "(cap=%s, %d/%d subjects at the cap)",
                 fit_source, n_fit.min(), float(np.median(n_fit)), n_fit.max(),
                 fit_kwargs["max_trials"], self.transform_fit_stats["n_at_cap"], n_fit.size)
        held_out = sorted(set(te["subject_id"].unique()) - set(tf_train))
        tf_test: Dict[int, SubjectTransform] = {}
        if held_out:
            cap = _get(cfg, "data.ea.test_subject_max_trials", None)
            kw = dict(fit_kwargs)
            if cap:
                kw["max_trials"] = int(cap)
            tf_test = fit_subject_transforms(
                te.loc[te["subject_id"].isin(held_out)], self.store,
                cache_path=self.run_dir / "subject_transforms_heldout.json", **kw,
            )
            log.info(
                "fit label-free transforms for %d held-out subjects from their own "
                "trials (cap=%s)", len(held_out), cap,
            )
        transforms = {**tf_train, **tf_test}

        common = dict(
            store=self.store, video=self.video, transforms=transforms,
            subject_index=self.subject_index, seed=self.seed,
            rescale=str(_get(cfg, "train.curriculum.rescale", "none")),
        )
        self.ds_train = TrialView(tr, **common)
        self.ds_val = TrialView(va, **common)
        self.ds_test = TrialView(te, **common)
        self.n_times = self.ds_train.n_times
        # kept for the step-matched arm: the batch count the *un-subsampled*
        # training split would have produced
        self._full_train_video_id = tr_full["video_id"].to_numpy(np.int64)
        self._full_train_subject = tr_full["subject_id"].to_numpy(np.int64)
        self.n_train_full = int(len(tr_full))
        log.info(
            "fold data: train=%d (%d subj, %d vids) val=%d (%d subj, %d vids) "
            "test=%d (%d subj, %d vids)",
            len(tr), tr["subject_id"].nunique(), tr["video_id"].nunique(),
            len(va), va["subject_id"].nunique(), va["video_id"].nunique(),
            len(te), te["subject_id"].nunique(), te["video_id"].nunique(),
        )
        self._record_data_budget(tr, va, te, tr_full)
        if len(va) == 0:
            raise RuntimeError(
                "inner validation split is empty -- early stopping would select on "
                "the test fold. Lower train.val.video_frac or change train.val.mode."
            )
        overlap = set(tr["video_id"]) & set(te["video_id"])
        if overlap and str(getattr(self.fold, "regime", "")) != "loso":
            log.warning("%d videos appear in both train and test", len(overlap))

    # -- trial-count scaling ------------------------------------------------- #

    def _subsample_train(self, tr: pd.DataFrame) -> tuple:
        """Apply ``train.train_trial_frac`` / ``train.n_train_trials`` to ``tr``.

        Training trials only.  ``train.trial_subsample.include_val: true`` extends
        it to the inner validation split as well -- off by default, because a
        validation set that shrinks with the fraction makes checkpoint selection
        noisier at exactly the points of the curve that are already noisiest, and
        that noise would be read as a data effect.
        """
        cfg = self.cfg
        frac = _get(cfg, "train.train_trial_frac", None)
        n_trials = _get(cfg, "train.n_train_trials", None)
        if frac is None and n_trials is None:
            return tr, {"applied": False, "reason": "not requested",
                        "n_before": int(len(tr)), "n_after": int(len(tr))}
        if frac is not None:
            frac = float(frac)
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"train.train_trial_frac must be in (0, 1], got {frac}")
        kept, info = stratified_trial_subsample(
            tr,
            frac=frac,
            n_trials=None if n_trials is None else int(n_trials),
            stratify=str(_get(cfg, "train.trial_subsample.stratify", "subject_condition")),
            unit=str(_get(cfg, "train.trial_subsample.unit", "trial")),
            seed=self.seed,
            salt=int(_get(cfg, "train.trial_subsample.seed_salt", 8821)),
            min_per_cell=int(_get(cfg, "train.trial_subsample.min_per_cell", 0)),
        )
        if info.get("applied"):
            log.info(
                "trial scaling: %d -> %d training trials (%.4f of the fold's train "
                "split, unit=%s stratify=%s); per-cell kept min/med/max = %d/%.1f/%d; "
                "%d subjects, %d videos, %d conditions still present",
                info["n_before"], info["n_after"], info["realised_frac"],
                info["unit"], info["stratify"],
                info["kept_per_cell_min"], info["kept_per_cell_median"],
                info["kept_per_cell_max"], info["n_subjects_after"],
                info["n_videos_after"], info["n_conditions_after"],
            )
            if info["n_videos_after"] < info["n_videos_before"]:
                log.warning(
                    "trial scaling dropped %d base video(s) from training -- the "
                    "distinct_video batch sampler will clamp the batch size, which "
                    "changes the n-way of the in-batch softmax. Use a larger "
                    "fraction or stratify=subject_condition.",
                    info["n_videos_before"] - info["n_videos_after"],
                )
        else:
            log.info("trial scaling: not applied (%s)", info.get("reason"))
        return kept, info

    def _record_data_budget(
        self, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, tr_full: pd.DataFrame
    ) -> None:
        """Write the evidence that the curve is a curve and not four experiments.

        ``eval_fingerprint.json`` pins the evaluation set (a sha1 over the raw
        bytes of the test ``trial_uid`` array, in order and as a set); it must be
        identical at every point of the scaling curve.  ``trial_subsample.json``
        records what was actually removed from training, and -- confound (b) --
        how many ``(subject, condition)`` cells can no longer supply the ``k``
        repeats the curriculum asks for.
        """
        k_start = int(_get(self.cfg, "train.curriculum.k_start", 4))
        k_eval = int(_get(self.cfg, "eval.pseudo_k", 4))
        self.cell_stats = {
            "train": cell_availability(tr, k_start),
            "train_full": cell_availability(tr_full, k_start),
            "val": cell_availability(va, int(_get(self.cfg, "train.val.pseudo_k", 4))),
            "test": cell_availability(te, k_eval),
        }
        self.eval_fingerprint = {
            "test": uid_fingerprint(te["trial_uid"].to_numpy(np.int64)),
            "val": uid_fingerprint(va["trial_uid"].to_numpy(np.int64)),
            "test_pseudo_units_k": k_eval,
            "test_pseudo_units": self.cell_stats["test"]["n_full_pseudo_trials"],
            "n_test_videos": int(te["video_id"].nunique()),
            "n_test_subjects": int(te["subject_id"].nunique()),
        }
        atomic_write_json(self.run_dir / "eval_fingerprint.json", self.eval_fingerprint)
        c = self.cell_stats["train"]
        if self.trial_subsample_applied:
            log.info(
                "pseudo-trial availability at k=%d after subsampling: %d/%d "
                "(subject, condition) cells below k (%.1f%% of anchors); realised "
                "mean k = %.2f (was %.2f on the full training split)",
                k_start, int(c["cells_below_k"]), int(c["n_cells"]),
                100 * c["frac_trials_below_k"], c["mean_k_avail"],
                self.cell_stats["train_full"]["mean_k_avail"],
            )
        self.budget_record: Dict[str, Any] = {
            "subsample": self.subsample_info,
            "n_train": int(len(tr)),
            "n_train_full": int(len(tr_full)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "cells": self.cell_stats,
            "transform_fit": self.transform_fit_stats,
            "eval_fingerprint": self.eval_fingerprint,
        }
        self._write_budget_record()

    def _write_budget_record(self) -> None:
        """(Re-)write ``trial_subsample.json``: everything the curve is read from."""
        rec = dict(getattr(self, "budget_record", {}))
        if hasattr(self, "batch_stats"):
            rec["batches"] = self.batch_stats
            rec["optimizer_steps_total"] = int(
                self.batch_stats["steps_per_epoch"] * int(_get(self.cfg, "train.epochs", 60))
                // max(1, int(_get(self.cfg, "train.accum_steps", 1)))
            )
        atomic_write_json(self.run_dir / "trial_subsample.json", rec)

    # -- model -------------------------------------------------------------- #

    def _build(self) -> None:
        cfg = self.cfg
        self.encoder, self.projector = build_model(
            _get(cfg, "model", {}) or {}, n_times=self.n_times, d_video=self.video.dim
        )
        self.loss_fn = build_loss_from_cfg(_get(cfg, "loss", {"name": "infonce"}))
        self.encoder.to(self.device)
        self.projector.to(self.device)
        self.loss_fn.to(self.device)
        self._declare_train_subjects()
        # A prototype bank carried across a video-disjoint fold boundary is
        # zero-shot leakage: the held-out videos' prototypes would still hold
        # the previous fold's EEG-derived estimates.  Zero it at every fold.
        self._reset_prototype_bank()

        wd = float(_get(cfg, "train.optimizer.weight_decay", 0.05))
        lr = float(_get(cfg, "train.optimizer.lr", 3e-4))

        # An encoder may define its own parameter groups.  A pretrained
        # foundation-model backbone needs a much smaller LR than a freshly
        # initialised projector, and expressing that as a per-group ``lr`` is the
        # only thing that works: AdamW is invariant to gradient scale, so hooks
        # that merely scale gradients do nothing.  Without this hand-off, an
        # encoder's ``backbone_lr_scale`` is silently inert and the only usable
        # fine-tuning control is freezing.
        own_groups = None
        pg_fn = getattr(self.encoder, "param_groups", None)
        if callable(pg_fn):
            try:
                own_groups = [g for g in pg_fn(lr, wd) if list(g.get("params", []))]
            except Exception as exc:  # never let a custom hook break training
                log.warning("encoder.param_groups(%g, %g) failed (%s); "
                            "falling back to a single learning rate", lr, wd, exc)
                own_groups = None

        mods = (self.projector,) if own_groups else (self.encoder, self.projector)
        decay, no_decay = [], []
        for mod in mods:
            for name, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
        loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if own_groups:
            groups = own_groups + groups
            log.info("encoder supplied %d custom parameter group(s); LRs: %s",
                     len(own_groups),
                     [round(float(g.get("lr", lr)), 8) for g in own_groups])
        if loss_params:  # learnable temperature must never be weight-decayed
            groups.append({"params": loss_params, "weight_decay": 0.0})
        opt_name = str(_get(cfg, "train.optimizer.name", "adamw")).lower()
        betas = tuple(_get(cfg, "train.optimizer.betas", [0.9, 0.98]))
        if opt_name == "adamw":
            self.opt = torch.optim.AdamW(groups, lr=lr, betas=betas)
        elif opt_name == "adam":
            self.opt = torch.optim.Adam(groups, lr=lr, betas=betas)
        elif opt_name == "sgd":
            self.opt = torch.optim.SGD(
                groups, lr=lr, momentum=float(_get(cfg, "train.optimizer.momentum", 0.9))
            )
        else:
            raise ValueError(f"unknown optimizer {opt_name!r}")

        # An encoder may carry provenance that belongs in the run record: which
        # pretrained checkpoint it loaded, what fraction of its backbone actually
        # came from that checkpoint, and what interface compromise was made to
        # fit our 120-sample epoch (padding fraction, patch size, channel map).
        # For the foundation-model rungs those facts ARE the caveat on the
        # number, so they must land in log.jsonl next to it, not only in a
        # docstring.
        desc_fn = getattr(self.encoder, "describe", None)
        if callable(desc_fn):
            with contextlib.suppress(Exception):
                self.encoder_description = desc_fn()
                # Do NOT truncate: the interface note (padding fraction, dropped
                # electrodes, sampling-rate mismatch) is the caveat on this rung's
                # number, and a [:2000] slice cut EEGPT's 2187-char record off
                # mid-word.  It also goes to log.jsonl as structured data so the
                # report can read it back rather than parse a log line.
                log.info("encoder provenance: %s",
                         json.dumps(self.encoder_description, default=str))

        n_par = sum(p.numel() for p in self.encoder.parameters()) + sum(
            p.numel() for p in self.projector.parameters()
        )
        log.info(
            "model: %s + projector, %.2fM params, loss=%s, device=%s, amp=%s",
            type(self.encoder).__name__, n_par / 1e6, type(self.loss_fn).__name__,
            self.device, self.amp_mode,
        )

        self.epochs = int(_get(cfg, "train.epochs", 60))
        self.target_steps_per_epoch = self._resolve_target_steps()
        self.train_loader = self._make_train_loader(epoch=0)
        self.steps_per_epoch = max(1, len(self.train_loader))
        sampler = self.train_loader.batch_sampler
        self.batch_stats = {
            "steps_per_epoch": int(self.steps_per_epoch),
            "target_steps_per_epoch": self.target_steps_per_epoch,
            "step_matching": getattr(self, "step_matching", "epoch_matched"),
            "batches_per_pass": int(getattr(sampler, "batches_per_pass", self.steps_per_epoch)),
            "passes_per_epoch": float(getattr(sampler, "passes_per_epoch", 1.0)),
            "batch_size": int(getattr(sampler, "batch_size", 0)),
            "requested_batch_size": int(_get(cfg, "train.batch.size", 72)),
            "n_train_videos": int(getattr(sampler, "n_videos", 0)),
            "n_train_trials": int(len(self.ds_train)),
        }
        log.info(
            "batches: %d/epoch (%s; %.2f pass(es) over %d training trials), "
            "batch=%d over %d distinct base videos",
            self.steps_per_epoch, self.batch_stats["step_matching"],
            self.batch_stats["passes_per_epoch"], self.batch_stats["n_train_trials"],
            self.batch_stats["batch_size"], self.batch_stats["n_train_videos"],
        )
        self.accum = max(1, int(_get(cfg, "train.accum_steps", 1)))
        total = max(1, (self.steps_per_epoch * self.epochs) // self.accum)
        warmup = int(float(_get(cfg, "train.scheduler.warmup_epochs", 2)) * self.steps_per_epoch / self.accum)
        min_factor = float(_get(cfg, "train.scheduler.min_lr_factor", 0.02))
        sched_name = str(_get(cfg, "train.scheduler.name", "cosine")).lower()

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / max(1, warmup)
            if sched_name == "constant":
                return 1.0
            prog = (step - warmup) / max(1, total - warmup)
            prog = min(max(prog, 0.0), 1.0)
            return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * prog))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    def _declare_train_subjects(self) -> None:
        """Tell the encoder which subjects it is allowed to have parameters for.

        This is not bookkeeping.  ``EEGEncoder.set_train_subjects`` is what makes a
        held-out subject fall back to the pre-registered unseen rule (mean token /
        identity adapter) instead of reading its own never-trained embedding row --
        which would quietly turn a LOSO number into a garbage-parameter number.  It
        must be re-declared before training *and* before evaluation, so it is
        called from ``_build``, ``validate`` and ``evaluate``.
        """
        fn = getattr(self.encoder, "set_train_subjects", None)
        if fn is None:
            return
        fn(sorted(int(s) for s in self.subject_index))
        state = getattr(self.encoder, "unseen_subject_state", None)
        if state is not None and not getattr(self, "_logged_unseen_rule", False):
            with contextlib.suppress(Exception):
                log.info("unseen-subject inference rule: %s", state())
            self._logged_unseen_rule = True

    # -- loaders ------------------------------------------------------------ #

    def _batch_sampler(
        self, video_id: np.ndarray, subject_id: np.ndarray, *, n_batches: int | None
    ) -> StratifiedVideoBatchSampler:
        cfg = self.cfg
        return StratifiedVideoBatchSampler(
            video_id,
            subject_id,
            batch_size=int(_get(cfg, "train.batch.size", 72)),
            mode=str(_get(cfg, "train.batch.mode", "distinct_video")),
            subjects_per_video=int(_get(cfg, "train.batch.subjects_per_video", 1)),
            drop_last=bool(_get(cfg, "train.batch.drop_last", True)),
            seed=self.seed,
            max_batches=_get(cfg, "train.max_steps_per_epoch", None),
            n_batches=n_batches,
        )

    def _resolve_target_steps(self) -> int | None:
        """``train.steps_per_epoch`` -> a batch count, or ``None`` for one pass.

        Confound (a) of the trial-scaling curve.  At a fixed ``train.epochs``,
        subsampling the trials also divides the number of optimizer steps -- and
        squeezes the warmup and the cosine decay into that smaller budget -- so an
        epoch-matched curve confuses "less data" with "less training".

        * ``null``      -- epoch-matched: one pass over the (subsampled) trials,
                           steps scale with the fraction.  Report it as such.
        * ``"full"``    -- step-matched: every fraction runs the number of batches
                           the *un-subsampled* training split of this fold would
                           have produced, by re-walking the smaller pool. The LR
                           schedule is then byte-identical across the curve.
        * ``<int>``     -- step-matched to an explicit budget (use this to match
                           steps across folds as well as across fractions).
        """
        spec = _get(self.cfg, "train.steps_per_epoch", None)
        if spec is None or (isinstance(spec, str) and spec.strip().lower() in
                            {"", "none", "null", "per_epoch", "epoch"}):
            self.step_matching = "epoch_matched"
            return None
        if isinstance(spec, str):
            if spec.strip().lower() not in {"full", "match_full", "auto"}:
                raise ValueError(
                    f"train.steps_per_epoch must be null, an int, or 'full'; got {spec!r}")
            full = self._batch_sampler(
                self._full_train_video_id, self._full_train_subject, n_batches=None
            )
            self.step_matching = "step_matched:full"
            log.info("step-matched arm: the full training split of this fold yields "
                     "%d batches/epoch; every fraction will run that many.", len(full))
            return len(full)
        self.step_matching = "step_matched:explicit"
        return max(1, int(spec))

    def _make_train_loader(self, epoch: int) -> DataLoader:
        cfg = self.cfg
        sampler = self._batch_sampler(
            self.ds_train.video_id, self.ds_train.subject,
            n_batches=getattr(self, "target_steps_per_epoch", None),
        )
        sampler.set_epoch(epoch)
        return DataLoader(
            self.ds_train,
            batch_sampler=sampler,
            num_workers=int(_get(cfg, "data.num_workers", 8)),
            pin_memory=bool(_get(cfg, "data.pin_memory", True)) and self.device.type == "cuda",
            collate_fn=collate,
            worker_init_fn=_worker_init,
            persistent_workers=False,  # the curriculum mutates dataset state
            prefetch_factor=int(_get(cfg, "data.prefetch_factor", 4))
            if int(_get(cfg, "data.num_workers", 8)) > 0 else None,
        )

    def _eval_loader(self, ds: Dataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=int(_get(self.cfg, "eval.batch_size", 512)),
            shuffle=False,
            num_workers=int(_get(self.cfg, "data.num_workers", 8)),
            pin_memory=self.device.type == "cuda",
            collate_fn=collate,
            worker_init_fn=_worker_init,
        )

    # -- curriculum --------------------------------------------------------- #

    def curriculum_state(self, epoch: int) -> tuple:
        """``(k, p_single)`` for ``epoch``.

        ``mode="prob_mix"`` (default) keeps ``k = k_start`` but raises the
        probability that any given anchor is a single trial from 0 to 1 over
        ``anneal_epochs``; the batch therefore contains a *mixture* throughout,
        which is a gentler distribution shift than a hard switch and leaves the
        last third of training purely single-trial.  ``mode="step"`` walks an
        explicit ``k`` ladder instead.
        """
        cfg = _get(self.cfg, "train.curriculum", {}) or {}
        if not bool(cfg.get("enabled", True)):
            return 1, 1.0
        k_start = int(cfg.get("k_start", 4))
        anneal = max(1, int(cfg.get("anneal_epochs", 12)))
        mode = str(cfg.get("mode", "prob_mix"))
        if mode == "step":
            ladder = list(cfg.get("ladder", [4, 4, 2, 2, 1]))
            i = min(int(epoch * len(ladder) / max(1, anneal)), len(ladder) - 1)
            return int(ladder[i]), 0.0
        p = min(1.0, epoch / float(anneal))
        return k_start, p

    # -- train -------------------------------------------------------------- #

    def fit(self) -> TrainResult:
        t0 = time.time()
        monitor = str(_get(self.cfg, "train.early_stop.monitor", "val/top1_pseudo"))
        patience = int(_get(self.cfg, "train.early_stop.patience", 10))
        min_delta = float(_get(self.cfg, "train.early_stop.min_delta", 0.0))
        val_every = max(1, int(_get(self.cfg, "train.val.every", 1)))
        status, message = "ok", ""

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            k, p_single = self.curriculum_state(epoch)
            self.ds_train.set_epoch(epoch, k, p_single)
            self.train_loader = self._make_train_loader(epoch)
            train_logs = self._train_one_epoch(epoch, k, p_single)
            self._sync_prototype_bank()   # DDP epoch boundary; no-op single-GPU

            row: Dict[str, Any] = {
                "epoch": epoch, "k": k, "p_single": round(p_single, 4),
                "lr": self.sched.get_last_lr()[0], **train_logs,
            }
            if (epoch + 1) % val_every == 0 or epoch == self.epochs - 1:
                val = self.validate()
                row.update(val)
                score = row.get(monitor, float("nan"))
                if not math.isfinite(score):
                    log.warning("monitor %s is not finite (%s); available: %s",
                                monitor, score, sorted(k for k in row if k.startswith("val/")))
                    score = -math.inf
                improved = score > self.best_monitor + min_delta
                if improved:
                    self.best_monitor, self.best_epoch = float(score), epoch
                    self.epochs_since_best = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.epochs_since_best += val_every
                row["best_monitor"] = self.best_monitor
                row["epochs_since_best"] = self.epochs_since_best

            self.history.append(row)
            self._append_log(row)
            self.save_checkpoint("last.pt")
            log.info(
                "[%s] epoch %d/%d k=%d loss=%.4f %s=%.4f (best %.4f @%d)",
                self.run_dir.name, epoch + 1, self.epochs, k,
                row.get("train/loss", float("nan")), monitor,
                row.get(monitor, float("nan")), self.best_monitor, self.best_epoch,
            )
            if patience > 0 and self.epochs_since_best >= patience:
                status, message = "early_stopped", f"no improvement for {self.epochs_since_best} epochs"
                log.info("early stop: %s", message)
                break

        metrics = self.evaluate()
        result = TrainResult(
            run_dir=str(self.run_dir),
            fold_key=self.run_dir.name,
            regime=str(getattr(self.fold, "regime", "")),
            status=status,
            best_epoch=self.best_epoch,
            best_monitor=self.best_monitor,
            epochs_run=self.epoch + 1,
            seconds=time.time() - t0,
            metrics=metrics,
            n_train=len(self.ds_train), n_val=len(self.ds_val), n_test=len(self.ds_test),
            n_train_subjects=int(len(self.subject_index)),
            n_test_videos=int(np.unique(self.ds_test.video_id).size),
            message=message,
            n_train_full=int(self.n_train_full),
            train_trial_frac=float(len(self.ds_train) / max(1, self.n_train_full)),
            steps_per_epoch=int(self.steps_per_epoch),
            optimizer_steps=int(self.global_step),
            step_matching=str(self.batch_stats["step_matching"]),
            passes_per_epoch=float(self.batch_stats["passes_per_epoch"]),
            train_batch_size=int(self.batch_stats["batch_size"]),
            n_train_videos=int(self.batch_stats["n_train_videos"]),
            cells_below_k=int(self.cell_stats["train"]["cells_below_k"]),
            frac_anchors_below_k=float(self.cell_stats["train"]["frac_trials_below_k"]),
            mean_k_avail=float(self.cell_stats["train"]["mean_k_avail"]),
            transform_fit_median=float(self.transform_fit_stats["n_fit_median"]),
            test_uid_sha1=str(self.eval_fingerprint["test"]["order_sha1"]),
            val_uid_sha1=str(self.eval_fingerprint["val"]["order_sha1"]),
        )
        atomic_write_json(self.run_dir / "metrics.json", result.to_dict())
        atomic_write_json(
            self.run_dir / "done.json",
            {"status": status, "finished": time.time(), "best_epoch": self.best_epoch},
        )
        return result

    # -- prototype-bank lifecycle (ProtoNCE and anything else bank-based) ---- #

    def _reset_prototype_bank(self) -> None:
        """Zero any prototype bank the loss owns.  No-op for bank-free losses."""
        fn = getattr(self.loss_fn, "reset_bank", None)
        if callable(fn):
            fn()
            log.info("prototype bank reset at fold boundary (%s)",
                     type(self.loss_fn).__name__)

    def _sync_prototype_bank(self) -> None:
        """Average prototype banks across DDP ranks at an epoch boundary."""
        fn = getattr(self.loss_fn, "sync_banks_", None)
        if callable(fn):
            fn()

    @torch.no_grad()
    def _install_zeroshot_prototypes(self, view: "TrialView") -> None:
        """Seed the bank with the projected frozen video embeddings of ``view``.

        Held-out videos have no EEG-derived prototype by construction, so
        without this the zero-shot gallery slots are all-zero and retrieval
        against them is meaningless.  The prototypes come from the *video*
        tower, never from test EEG.
        """
        fn = getattr(self.loss_fn, "set_prototypes", None)
        if not callable(fn):
            return
        grans = list(getattr(self.loss_fn, "granularities", []) or [])
        if not grans:
            return
        self.projector.eval()
        for gran in grans:
            # contract C indexing: cond_emb[condition_id], base_emb[video_id - 1]
            if gran == "condition":
                uniq = np.unique(view.condition_id)
                emb = self.video.cond_emb[uniq]
            else:
                uniq = np.unique(view.video_id)
                emb = self.video.base_emb[uniq - 1]
            z = self.projector(torch.as_tensor(emb, dtype=torch.float32, device=self.device))
            fn(torch.as_tensor(uniq, device=self.device), z, granularity=gran)
        cov = getattr(self.loss_fn, "bank_coverage", None)
        log.info("installed zero-shot prototypes for %s; bank coverage=%s",
                 grans, cov() if callable(cov) else "n/a")

    def _train_one_epoch(self, epoch: int, k: int, p_single: float) -> Dict[str, float]:
        self.encoder.train()
        self.projector.train()
        self.loss_fn.train()
        clip = float(_get(self.cfg, "train.grad_clip", 1.0))
        log_every = int(_get(self.cfg, "train.log_every", 50))
        agg: Dict[str, float] = {}
        n_batches = 0
        t0 = time.time()

        self.opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            x = batch["x"].to(self.device, non_blocking=True)
            v = batch["v"].to(self.device, non_blocking=True)
            sidx = batch["subject_index"].to(self.device, non_blocking=True)
            meta = {mk: batch[mk].to(self.device, non_blocking=True) for mk in META_KEYS if mk in batch}

            ctx = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                if self.use_amp else contextlib.nullcontext()
            )
            with ctx:
                z_eeg = self.encoder(x, sidx)
                z_vid = self.projector(v)
            # the contrastive objective itself always runs in fp32: a softmax over
            # 72 near-collinear cosines is exactly where fp16 loses its resolution
            z_eeg = _renorm(z_eeg.float())
            z_vid = _renorm(z_vid.float())
            out = self.loss_fn(z_eeg, z_vid, meta)
            loss = out["loss"] if isinstance(out, Mapping) else out
            if not torch.isfinite(loss):
                log.error("non-finite loss at epoch %d step %d -- skipping batch", epoch, step)
                self.opt.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss / self.accum).backward()
            if (step + 1) % self.accum == 0:
                if clip and clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.opt.param_groups for p in g["params"]], clip
                    )
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                self.sched.step()
                # CompositeLoss keeps its own step counter for warmup_steps /
                # start_step ramps and it does NOT advance itself.  Without this
                # call the counter stays at 0 forever, so every component with a
                # warmup has effective weight 0.0 for the entire run and
                # contributes nothing -- silently, because the term is still
                # computed and still logged.  Measured on atm_composite.yaml:
                # clisa (weight 0.2, warmup_steps 500) logged weight = 0.0 at
                # every epoch, which nulls the lambda_1 redundancy ablation that
                # config exists to run.
                step_fn = getattr(self.loss_fn, "step", None)
                if callable(step_fn):
                    step_fn()
                self.global_step += 1

            agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
            if "k" in batch:
                # the k the curriculum ASKED for is logged as `k`; this is the k it
                # actually got.  They diverge whenever a (subject, condition) cell
                # holds fewer than k trials -- which is exactly what a trial
                # subsample causes, so on a scaling curve the two must be read
                # together or a change of objective reads as a change of data.
                agg["k_realised"] = agg.get("k_realised", 0.0) + float(batch["k"].float().mean())
            if isinstance(out, Mapping):
                for lk, lv in (out.get("logs", {}) or {}).items():
                    with contextlib.suppress(TypeError, ValueError):
                        agg[lk] = agg.get(lk, 0.0) + float(lv)
            n_batches += 1
            if log_every and step % log_every == 0:
                log.debug("  e%d s%d loss=%.4f", epoch, step, float(loss.detach()))

        n = max(1, n_batches)
        out = {f"train/{a}": v / n for a, v in agg.items()}
        out["train/steps"] = float(n_batches)
        out["train/seconds"] = time.time() - t0
        return out

    # -- eval --------------------------------------------------------------- #

    @torch.no_grad()
    def encode(self, ds: Dataset) -> Dict[str, np.ndarray]:
        """Embed every item of ``ds``; returns z plus the meta columns."""
        self.encoder.eval()
        self.projector.eval()
        zs, cols = [], {k: [] for k in ("row", "trial_uid", "video_id", "condition_id",
                                        "subject_id", "material_id", "orientation")}
        for batch in self._eval_loader(ds):
            x = batch["x"].to(self.device, non_blocking=True)
            sidx = batch["subject_index"].to(self.device, non_blocking=True)
            ctx = (
                torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                if self.use_amp else contextlib.nullcontext()
            )
            with ctx:
                z = self.encoder(x, sidx)
            zs.append(_renorm(z.float()).cpu().numpy())
            for c in cols:
                if c in batch:
                    cols[c].append(batch[c].cpu().numpy())
        out = {"z": np.concatenate(zs) if zs else np.zeros((0, 1), np.float32)}
        for c, v in cols.items():
            if v:
                out[c] = np.concatenate(v)
        return out

    @torch.no_grad()
    def project_gallery(self, unit: str, ids: np.ndarray) -> np.ndarray:
        """Project the frozen video embeddings of ``ids`` into the shared space.

        ``unit="video"`` -> ``ids`` are ``video_id`` (1..90) and the *original*
        orientation embedding is used; ``unit="condition"`` -> ``condition_id``.
        """
        self.projector.eval()
        src = self.video.base_emb[np.asarray(ids) - 1] if unit == "video" \
            else self.video.cond_emb[np.asarray(ids)]
        t = torch.from_numpy(np.ascontiguousarray(src, dtype=np.float32)).to(self.device)
        z = self.projector(t)
        return _renorm(z.float()).cpu().numpy()

    def _retrieval_block(
        self,
        enc: Mapping[str, np.ndarray],
        unit: str,
        prefix: str,
        gallery_sizes: Sequence[int],
        n_draws: int,
        with_per_subject: bool = False,
    ) -> tuple:
        """Full-gallery + subsampled-gallery retrieval for one embedding set."""
        key = enc["video_id"] if unit == "video" else enc["condition_id"]
        gal_ids = np.unique(key)
        if gal_ids.size < 2:
            return {}, {}
        pos = {int(g): i for i, g in enumerate(gal_ids)}
        true_idx = np.array([pos[int(v)] for v in key], dtype=np.int64)
        gal = self.project_gallery(unit, gal_ids)
        sim = enc["z"] @ gal.T

        groups = None
        if unit == "video" and "material_id" in enc:
            m = {int(v): int(g) for v, g in zip(key, enc["material_id"])}
            groups = np.array([m.get(int(g), -1) for g in gal_ids])
        full = self.retrieval_metrics(enc["z"], gal, true_idx, groups)
        out = {f"{prefix}/{unit}/full/{k}": float(v) for k, v in full.items()
               if isinstance(v, (int, float))}
        out[f"{prefix}/{unit}/gallery_size"] = float(gal_ids.size)

        rng = np.random.default_rng([self.seed, 4242])
        for size in gallery_sizes:
            if size < 2 or size > gal_ids.size:
                continue
            m = _subsampled_metrics(sim, true_idx, int(size), n_draws, rng)
            for k, v in m.items():
                out[f"{prefix}/{unit}/g{int(size)}/{k}"] = float(v)

        per_subject: Dict[int, Dict[str, float]] = {}
        if with_per_subject and "subject_id" in enc:
            true_score = sim[np.arange(sim.shape[0]), true_idx]
            rank = 1 + (sim > true_score[:, None]).sum(axis=1)
            for s in np.unique(enc["subject_id"]):
                m = enc["subject_id"] == s
                per_subject[int(s)] = {
                    f"{prefix}/{unit}/full/top1": float((rank[m] == 1).mean()),
                    f"{prefix}/{unit}/full/top5": float((rank[m] <= 5).mean()),
                    f"{prefix}/{unit}/full/mean_rank": float(rank[m].mean()),
                    "n": int(m.sum()),
                }
        return out, per_subject

    def validate(self) -> Dict[str, float]:
        """Inner-split retrieval used for early stopping and checkpoint choice."""
        self._declare_train_subjects()
        cfg = self.cfg
        unit = str(_get(cfg, "train.val.gallery_unit", "video"))
        sizes = list(_get(cfg, "train.val.gallery_sizes", []) or [])
        n_draws = int(_get(cfg, "train.val.n_resamples", 5))
        out: Dict[str, float] = {}
        single, _ = self._retrieval_block(self.encode(self.ds_val), unit, "val", sizes, n_draws)
        out.update({k.replace(f"/{unit}/full/", "/"): v for k, v in single.items()
                    if "/full/" in k})
        k_pseudo = int(_get(cfg, "train.val.pseudo_k", 4))
        if k_pseudo > 1:
            pv = PseudoTrialView(self.ds_val, k=k_pseudo, resample=0, seed=self.seed)
            if len(pv) >= 2:
                ps, _ = self._retrieval_block(self.encode(pv), unit, "val", sizes, n_draws)
                out.update({k.replace(f"/{unit}/full/", "/") + "_pseudo": v
                            for k, v in ps.items() if "/full/" in k})
        return out

    def evaluate(self, checkpoint: str = "best.pt") -> Dict[str, float]:
        """Final test-fold evaluation from the selected checkpoint."""
        ckpt = self.run_dir / "checkpoints" / checkpoint
        if ckpt.exists():
            self.load_checkpoint(ckpt, strict_optimizer=False)
            log.info("evaluating from %s (epoch %d)", checkpoint, self.best_epoch)
        else:
            log.warning("%s not found; evaluating the final weights", ckpt)
        self._declare_train_subjects()  # a reloaded checkpoint does not carry it
        # Held-out videos were never seen in training, so their bank slots are
        # uninitialised.  Install the projected frozen video embeddings as the
        # zero-shot gallery before scoring.
        self._install_zeroshot_prototypes(self.ds_test)

        cfg = self.cfg
        sizes = list(_get(cfg, "eval.gallery_sizes", [2, 10, 18, 90]) or [])
        units = list(_get(cfg, "eval.gallery_units", ["video", "condition"]) or [])
        n_draws = int(_get(cfg, "eval.n_resamples", 20))
        k_pseudo = int(_get(cfg, "eval.pseudo_k", 4))
        n_res = int(_get(cfg, "eval.pseudo_resamples", 5))

        metrics: Dict[str, float] = {}
        per_subject_rows: List[Dict[str, Any]] = []
        enc = self.encode(self.ds_test)
        for unit in units:
            m, per_sub = self._retrieval_block(enc, unit, "test", sizes, n_draws, True)
            metrics.update(m)
            for sid, d in per_sub.items():
                row = {"subject_id": sid, "trial_type": "single", "unit": unit}
                row.update({k.split("/")[-1]: v for k, v in d.items() if k != "n"})
                row["n"] = d["n"]
                per_subject_rows.append(row)

        if k_pseudo > 1:
            acc: Dict[str, List[float]] = {}
            per_sub_acc: Dict[tuple, List[Dict[str, float]]] = {}
            for r in range(max(1, n_res)):
                pv = PseudoTrialView(self.ds_test, k=k_pseudo, resample=r, seed=self.seed)
                if len(pv) < 2:
                    break
                penc = self.encode(pv)
                for unit in units:
                    m, per_sub = self._retrieval_block(penc, unit, "test", sizes, n_draws, True)
                    for k, v in m.items():
                        acc.setdefault(k + "_pseudo", []).append(v)
                    for sid, d in per_sub.items():
                        per_sub_acc.setdefault((sid, unit), []).append(d)
            metrics.update({k: float(np.mean(v)) for k, v in acc.items()})
            metrics.update({k + "_sd": float(np.std(v)) for k, v in acc.items() if len(v) > 1})
            for (sid, unit), ds in per_sub_acc.items():
                row = {"subject_id": sid, "trial_type": f"pseudo_k{k_pseudo}", "unit": unit}
                keys = [k for k in ds[0] if k != "n"]
                row.update({k.split("/")[-1]: float(np.mean([d[k] for d in ds])) for k in keys})
                row["n"] = int(np.mean([d["n"] for d in ds]))
                per_subject_rows.append(row)

        if per_subject_rows:
            pd.DataFrame(per_subject_rows).to_csv(self.run_dir / "per_subject.csv", index=False)
        if bool(_get(cfg, "eval.save_embeddings", True)):
            np.savez_compressed(
                self.run_dir / "test_embeddings.npz",
                z_eeg=enc["z"].astype(np.float32),
                trial_uid=enc.get("trial_uid"),
                video_id=enc.get("video_id"),
                condition_id=enc.get("condition_id"),
                subject_id=enc.get("subject_id"),
                gallery_video_ids=np.unique(enc["video_id"]),
                z_vid_video=self.project_gallery("video", np.unique(enc["video_id"])),
                gallery_condition_ids=np.unique(enc["condition_id"]),
                z_vid_condition=self.project_gallery("condition", np.unique(enc["condition_id"])),
            )
        atomic_write_json(self.run_dir / "test_metrics.json", metrics)
        return metrics

    # -- checkpoints -------------------------------------------------------- #

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
            "loss": self.loss_fn.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_monitor": self.best_monitor,
            "best_epoch": self.best_epoch,
            "epochs_since_best": self.epochs_since_best,
            "history": self.history,
            "cont_stats": self.cont_stats,
            "subject_index": self.subject_index,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "cfg": self.cfg,
        }

    def save_checkpoint(self, name: str = "last.pt") -> Path:
        path = self.run_dir / "checkpoints" / name
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)  # atomic: a kill mid-save cannot corrupt last.pt
        return path

    def load_checkpoint(self, path: Path | str, strict_optimizer: bool = True) -> None:
        path = Path(path)
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(blob["encoder"])
        self.projector.load_state_dict(blob["projector"])
        with contextlib.suppress(Exception):
            self.loss_fn.load_state_dict(blob["loss"])
        if strict_optimizer:
            self.opt.load_state_dict(blob["opt"])
            self.sched.load_state_dict(blob["sched"])
            with contextlib.suppress(Exception):
                self.scaler.load_state_dict(blob["scaler"])
            self.epoch = int(blob.get("epoch", -1)) + 1
            self.global_step = int(blob.get("global_step", 0))
            self.history = list(blob.get("history", []))
            rng = blob.get("rng") or {}
            with contextlib.suppress(Exception):
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
        self.best_monitor = float(blob.get("best_monitor", -math.inf))
        self.best_epoch = int(blob.get("best_epoch", -1))
        self.epochs_since_best = int(blob.get("epochs_since_best", 0))
        log.info("loaded checkpoint %s (resuming at epoch %d)", path.name, self.epoch)

    def maybe_resume(self) -> bool:
        last = self.run_dir / "checkpoints" / "last.pt"
        if last.exists():
            try:
                self.load_checkpoint(last, strict_optimizer=True)
                return True
            except Exception as exc:
                log.error("could not resume from %s (%s); starting fresh", last, exc)
        return False

    def _append_log(self, row: Mapping[str, Any]) -> None:
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: _jsonable(v) for k, v in row.items()}) + "\n")


def _renorm(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# --------------------------------------------------------------------------- #
# inner validation split
# --------------------------------------------------------------------------- #


def _stratified_video_sample(
    trials: pd.DataFrame, frac: float, seed: int, stratify: str = "material"
) -> np.ndarray:
    """Sample a fraction of base videos, stratified by material (8 classes).

    Material is the only attribute whose class count divides 90 tolerably;
    touch_type (12 classes) cannot be stratified across folds at all, which is
    why it is not an option here.
    """
    rng = np.random.default_rng([seed, 991])
    per_video = trials.drop_duplicates(subset=["video_id"])
    picked: List[int] = []
    if stratify in per_video.columns:
        for _, grp in per_video.groupby(stratify, sort=True, observed=True):
            vids = np.sort(grp["video_id"].to_numpy())
            n = int(max(1, round(frac * vids.size))) if frac > 0 else 0
            if n:
                picked.extend(rng.choice(vids, size=min(n, vids.size), replace=False).tolist())
    else:
        vids = np.sort(per_video["video_id"].to_numpy())
        n = int(max(1, round(frac * vids.size))) if frac > 0 else 0
        picked = rng.choice(vids, size=min(n, vids.size), replace=False).tolist()
    return np.sort(np.asarray(picked, dtype=np.int64))


def _inner_val_split(
    trials: pd.DataFrame, *, mode: str, video_frac: float, subject_frac: float, seed: int
) -> tuple:
    """Carve a validation split out of the *training* trials.

    ``mode``:

    ``video``
        Hold out a material-stratified set of training videos (all subjects).
    ``video_subject``
        Also hold out a set of training subjects, and drop the cross cells, so
        the inner split generalizes in both directions -- the same shape as the
        double-disjoint test fold it is standing in for.
    ``trial``
        A plain random trial split.  Leaky by construction (the same video is in
        both halves); provided only for debugging a pipeline end-to-end.
    """
    if mode == "trial":
        rng = np.random.default_rng([seed, 12345])
        m = rng.random(len(trials)) < video_frac
        return trials.loc[~m].reset_index(drop=True), trials.loc[m].reset_index(drop=True)

    val_videos = _stratified_video_sample(trials, video_frac, seed)
    in_val_v = trials["video_id"].isin(val_videos)
    if mode == "video_subject" and subject_frac > 0:
        rng = np.random.default_rng([seed, 553])
        subs = np.sort(trials["subject_id"].unique())
        n = int(max(1, round(subject_frac * subs.size)))
        val_subs = np.sort(rng.choice(subs, size=min(n, max(1, subs.size - 1)), replace=False))
        in_val_s = trials["subject_id"].isin(val_subs)
        val = trials.loc[in_val_v & in_val_s]
        train = trials.loc[~in_val_v & ~in_val_s]
    else:
        val = trials.loc[in_val_v]
        train = trials.loc[~in_val_v]
    return train.reset_index(drop=True), val.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# fold entry point
# --------------------------------------------------------------------------- #


def train_fold(
    cfg: Any,
    fold: Any,
    trials: pd.DataFrame,
    run_dir: Path | str,
    *,
    device: str | torch.device | None = None,
    resume: bool = True,
    force: bool = False,
    store: EpochStore | None = None,
    video: VideoEmbeddings | None = None,
) -> TrainResult:
    """Train + evaluate one fold, skipping it if it already finished.

    Idempotency is by ``done.json``: an unattended sweep can be killed and
    relaunched with the same command line and it will pick up where it stopped
    (finished folds skipped, the interrupted fold resumed from ``last.pt``).
    """
    run_dir = Path(run_dir)
    done = run_dir / "done.json"
    if done.exists() and not force:
        with open(done, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        metrics = {}
        mpath = run_dir / "test_metrics.json"
        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)
        log.info("skipping %s (already done)", run_dir.name)
        return TrainResult(
            run_dir=str(run_dir), fold_key=run_dir.name,
            regime=str(getattr(fold, "regime", "")), status="skipped",
            best_epoch=int(info.get("best_epoch", -1)), metrics=metrics,
            message="done.json present",
        )
    if force:
        for name in ("done.json", "metrics.json", "test_metrics.json"):
            with contextlib.suppress(FileNotFoundError):
                (run_dir / name).unlink()

    trainer = Trainer(cfg, fold, trials, run_dir, device=device, store=store, video=video)
    if resume and not force:
        trainer.maybe_resume()
    return trainer.fit()
