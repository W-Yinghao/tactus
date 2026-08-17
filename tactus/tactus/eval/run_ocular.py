#!/usr/bin/env python
"""Driver for the ocular half of gate G6.

:mod:`tactus.eval.probes` owns the probes; it has no ``__main__`` and no way to
reach the data.  This module is the missing driver for G6's second criterion:

    "the pre-saccadic (< 150 ms) signal survives frontal-channel ablation, and
     the EEG model beats an EOG-surrogate baseline"

It loads ``trials.parquet`` + the per-subject epoch memmaps + a frozen video
embedding bank, builds **video-disjoint** folds with
:func:`tactus.data.splits.make_folds`, and reports three things on identical
splits and one shared fold.

Two things this driver has to get right, both of which it previously got wrong
and both of which independently destroy the answer:

**The gallery is the fold's held-out base videos, not all 90.**
``n_video_folds=5`` over 90 base videos leaves 18 held out -- "18-way" *is* the
held-out set, the same construction as
:func:`tactus.baselines.linear_align._gallery_for`, which builds the gallery
from the unique ``video_id`` of the test meta.  Handing the whole 90-video bank
to :func:`~tactus.eval.retrieval.evaluate_retrieval` still produces rows
labelled ``gallery=nway18`` -- an analytic draw of 17 distractors of which ~14
were in the ridge's training set -- and :mod:`tactus.eval.report` merges on
exactly ``(gallery, metric, trial_type)``, so those rows would be tabulated next
to the deep-model numbers as if they measured the same thing.  The **training
targets** still come from the full 90-video bank (a training trial's target is
its own video's embedding); only the retrieval gallery is restricted.

**Both prediction variants are scored** (:data:`~tactus.eval.probes.PREDICTION_VARIANTS`).
The fitted intercept is a query-independent constant that dominates the
EEG-dependent term on this anisotropic SigLIP2 gallery and pins top-1 to chance;
scoring only ``with_mean`` returns "no signal" for every possible input, which
is indistinguishable from the confound being absent.  See
:meth:`tactus.baselines.linear_align.RidgeAlpha.predict`, which documents the
same failure for the linear baseline.

Reports, on identical splits and one shared fold:

(a) alignment with all 64 channels vs with the eight ocular channels ablated
    (:data:`~tactus.eval.probes.OCULAR_ABLATION_CHANNELS`, decision D6 -- the
    explicit list, not prefix matching, which removed 26/64 channels);
(b) an EOG-surrogate-only baseline the EEG arm must beat.  Both constructions of
    the same HEOG/VEOG definition are run.  ``ocular_surrogate`` is rebuilt by
    :func:`~tactus.eval.probes.ocular_surrogates` from the **robust-scaled**
    epochs; ``eog_surrogate_saved`` reads
    ``derived/ica/sub-XX_<window>_eogproxy.npy``, which
    :mod:`tactus.data.preprocess` derived from the **unscaled** continuous data.
    These are not the same signal: per-channel robust scaling does not commute
    with the F7-F8 difference (sub-01 scales F7 = 18.36 uV, F8 = 16.13 uV, a
    1.14x ratio), so the rebuilt HEOG is a *reweighted* difference.  Measured on
    sub-01 the two correlate r = 0.995 (HEOG) / 0.998 (VEOG), i.e. close but not
    identical; the saved proxy is the physically faithful one and the rebuilt one
    is what the probe API produces by default.  Reporting both keeps that visible;
(c) the same comparison split by time window: pre-saccadic (0-150 ms) vs
    sustained (150 ms-end).  Both sides are run because the pre-saccadic claim
    is a *contrast*, not a single number.

Positive control (``--positive-control``, on by default)
-------------------------------------------------------
A confound battery whose every arm lands at chance is un-diagnosable: it looks
identical whether the confound is absent, the data are misaligned, or the probe
is broken.  So the same feature map, same folds and same channel subsets are
also used to decode **orientation** -- a label this dataset is known to carry
(STATUS.md: +4.8 pp over the empirical floor with the linear MVPA baseline).
If orientation decodes and the alignment endpoint does not, the null is the
probe's, not the pipeline's.  This is an instrument check; it is **not** the
gate endpoint and must never be reported as one.

Every result carries :data:`~tactus.eval.probes.OCULAR_CLAIM_LIMITATION`
verbatim.

CLI
---
    python -m tactus.eval.run_ocular --subjects 1-8 --fold 0 \\
        --out $TACTUS_RESULTS_DIR/ocular
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data import (
    CHANNELS_JSON,
    DEFAULT_WINDOW,
    EPOCH_DIR,
    ICA_DIR,
    TRIALS_PARQUET,
    VIDEO_EMB_DIR,
    epoch_path,
    window_spec,
)
from ..data.splits import make_folds
from .probes import (
    OCULAR_ABLATION_CHANNELS,
    OCULAR_CLAIM_LIMITATION,
    PREDICTION_VARIANTS,
    PRIMARY_PREDICTION_VARIANT,
    PRIMARY_PROBE_ENDPOINT,
    RidgeAlignmentProbe,
    _arm_scores,
    _resolve_trial_type,
    arm_significance,
    frontal_ablation_sensitivity,
    ocular_control,
    select_channels,
)
from .retrieval import RetrievalConfig

__all__ = [
    "parse_subjects",
    "load_video_bank",
    "load_gallery",
    "load_channels",
    "TrialSource",
    "run_ocular_battery",
    "g6_criterion2_verdict",
    "mergeable_table",
    "summary_frame",
    "main",
]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def parse_subjects(spec: str) -> List[int]:
    """``"1-3,7"`` -> ``[1, 2, 3, 7]``."""
    out: List[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def load_channels(path: Optional[Path] = None) -> List[str]:
    payload = json.loads(Path(path or CHANNELS_JSON).read_text(encoding="utf-8"))
    names = payload.get("channels") or payload["ch_names"]
    return [str(c) for c in names]


def load_video_bank(emb_tag: str, root: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(base_emb (90, D), video_ids (90,))`` -- the **full** base-video bank.

    This is the training-target lookup, not the retrieval gallery: a training
    trial's ridge target is the embedding of its own base video, and those videos
    are by construction absent from the held-out gallery.  ``cond_emb`` would be
    the 360-way condition bank and is a different endpoint.
    """
    npz_path = Path(root or VIDEO_EMB_DIR) / f"{emb_tag}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"video embedding bank {npz_path} not found; available: "
            f"{sorted(p.stem for p in Path(root or VIDEO_EMB_DIR).glob('*.npz'))}"
        )
    blob = np.load(npz_path)
    base = np.asarray(blob["base_emb"], dtype=np.float64)
    return base, np.arange(1, base.shape[0] + 1, dtype=np.int64)


def load_gallery(
    emb_tag: str,
    root: Optional[Path] = None,
    *,
    video_ids: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(emb (M, D), video_ids (M,))`` -- the **retrieval** gallery.

    ``video_ids`` is the fold's held-out base videos.  It is not optional in
    practice: the pre-registered endpoint is 18-way retrieval among the videos
    the model never saw, so the gallery is exactly those 18 items and
    ``nway18_top1`` is then the full-gallery top-1 (the analytic N-way formula is
    exact when ``N == M``).  Passing ``None`` returns all 90 and is kept only for
    the explicitly-labelled diagnostic scope in :func:`run_ocular_battery`; on
    that gallery a ``nway18`` row means "17 distractors drawn from 90, most of
    them seen in training", which is a different -- and much easier -- endpoint
    wearing the same column label.

    The ids are returned sorted and de-duplicated so the gallery order is a
    deterministic function of the fold, not of trial order.
    """
    base, all_ids = load_video_bank(emb_tag, root)
    if video_ids is None:
        return base, all_ids
    want = np.unique(np.asarray(list(video_ids), dtype=np.int64))
    if want.size == 0:
        raise ValueError("empty gallery: no held-out video ids were supplied")
    missing = want[(want < 1) | (want > base.shape[0])]
    if missing.size:
        raise KeyError(
            f"video ids {missing.tolist()} are outside the embedding bank "
            f"(1..{base.shape[0]})"
        )
    return base[want - 1], want


class TrialSource:
    """Row-aligned access to the epoch memmaps and the saved EOG proxies.

    ``trial_uid`` is global; the memmap row is ``within_subj_idx``.  Both are
    read off the trial table rather than recomputed, so a change to the uid
    convention cannot silently scramble the gather.
    """

    def __init__(
        self,
        trials: pd.DataFrame,
        subjects: Sequence[int],
        window: str,
        *,
        epoch_dir: Optional[Path] = None,
        ica_dir: Optional[Path] = None,
    ) -> None:
        self.trials = trials
        self.index = trials.set_index("trial_uid")
        self.window = window
        self.epochs: Dict[int, np.ndarray] = {}
        self.eog: Dict[int, np.ndarray] = {}
        edir = Path(epoch_dir or EPOCH_DIR)
        idir = Path(ica_dir or ICA_DIR)
        for s in subjects:
            ep = epoch_path(int(s), window, edir)
            if not ep.exists():
                raise FileNotFoundError(f"missing epoch memmap {ep}")
            self.epochs[int(s)] = np.load(ep, mmap_mode="r")
            proxy = idir / f"sub-{int(s):02d}_{window}_eogproxy.npy"
            if not proxy.exists():
                raise FileNotFoundError(f"missing saved EOG proxy {proxy}")
            self.eog[int(s)] = np.load(proxy, mmap_mode="r")

    def gather(self, uids: np.ndarray, *, which: str = "eeg") -> Tuple[np.ndarray, pd.DataFrame]:
        rows = self.index.loc[np.asarray(uids)]
        store = self.epochs if which == "eeg" else self.eog
        subs = rows["subject_id"].to_numpy(dtype=np.int64)
        pos = rows["within_subj_idx"].to_numpy(dtype=np.int64)
        n_ch, n_t = store[int(subs[0])].shape[1:]
        out = np.empty((len(rows), n_ch, n_t), dtype=np.float32)
        for s in np.unique(subs):
            sel = np.flatnonzero(subs == s)
            out[sel] = store[int(s)][pos[sel]]
        return out, rows


# --------------------------------------------------------------------------- #
# extra arms the probes module does not own
# --------------------------------------------------------------------------- #
#: The G5 linear-MVPA orientation result recorded in STATUS.md, kept here so the
#: positive control can be reconciled against it in the same table rather than in
#: a reader's head.  Balanced accuracy, single timepoint, ``--cv sequence``, all
#: 80 subjects; peak 0.298 at 100 ms against a balanced chance of 0.250.
STATUS_MD_MVPA_ORIENTATION: Dict[str, Any] = {
    "above_floor_pp": 4.8,
    "estimator": "single-timepoint balanced accuracy (peak 0.298 vs 0.250 @ 100 ms)",
    "cv": "sequence",
    "n_subjects": 80,
    "source": "STATUS.md, section 'G5 线性 MVPA'",
}


def _windows_for(times_ms: np.ndarray, presaccadic_ms: float) -> List[Tuple[str, Optional[np.ndarray]]]:
    pre = np.flatnonzero(times_ms < presaccadic_ms)
    sus = np.flatnonzero(times_ms >= presaccadic_ms)
    out: List[Tuple[str, Optional[np.ndarray]]] = [("", None)]
    if pre.size >= 3:
        out.append(("_presaccadic", pre))
    if sus.size >= 3:
        out.append(("_sustained", sus))
    return out


def saved_eog_arms(
    eog_train: np.ndarray,
    y_train: np.ndarray,
    eog_test: np.ndarray,
    test_item_id: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_item_ids: np.ndarray,
    *,
    times_ms: np.ndarray,
    presaccadic_ms: float,
    groups: Optional[np.ndarray],
    subject_ids_test: Optional[np.ndarray],
    cfg: Optional[RetrievalConfig],
    probe_factory,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """The `eog_surrogate_saved` arms, from ``derived/ica/*_eogproxy.npy``.

    Identical pipeline to every other arm -- same probe, same folds, same
    gallery, both prediction variants -- so the numbers sit on the same footing.
    """
    frames: List[pd.DataFrame] = []
    diags: Dict[str, Dict[str, float]] = {}
    for suffix, keep_t in _windows_for(times_ms, presaccadic_ms):
        a_tr = eog_train if keep_t is None else eog_train[:, :, keep_t]
        a_te = eog_test if keep_t is None else eog_test[:, :, keep_t]
        probe = probe_factory()
        frames.append(
            probe.fit_evaluate(
                a_tr, y_train, a_te, test_item_id, gallery_emb, gallery_item_ids,
                groups=groups, subject_ids_test=subject_ids_test, cfg=cfg,
                tag=f"eog_surrogate_saved{suffix}",
            )
        )
        diags[f"eog_surrogate_saved{suffix}"] = dict(probe.diagnostics_)
    return pd.concat(frames, ignore_index=True), diags


def orientation_positive_control(
    x_train: np.ndarray,
    x_test: np.ndarray,
    rows_train: pd.DataFrame,
    rows_test: pd.DataFrame,
    channel_names: Sequence[str],
    *,
    times_ms: np.ndarray,
    presaccadic_ms: float,
    eog_train: np.ndarray,
    eog_test: np.ndarray,
    probe_factory,
    drop_channels: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Instrument check: can the SAME features decode orientation?

    Reported against the **empirical majority-class floor**, not against
    ``1/n_classes`` -- STATUS.md records that the uniform-chance convention made
    a majority-class predictor look far above chance on this dataset.

    Not the gate endpoint.  Its only job is to distinguish "no ocular confound"
    from "no working probe".

    The number it produces is **not** comparable to the G5 MVPA number without
    saying so.  This is a whole-window estimator: 64 channels x
    ``n_time_bins`` pooled bins (1280 features) into one multinomial logistic
    fit, scored with plain accuracy against the empirical floor.  STATUS.md's
    ``+4.8 pp`` is a *single-timepoint* balanced-accuracy peak (0.298 vs 0.250 at
    100 ms) over all 80 subjects with ``--cv sequence``.  A whole-window
    estimator integrates every timepoint's evidence, so it should be larger; the
    gap is reported explicitly as ``status_md_ratio`` rather than left for a
    reader to trip over.  See :data:`STATUS_MD_MVPA_ORIENTATION`.
    """
    from sklearn.linear_model import LogisticRegression

    ablate = set(drop_channels or OCULAR_ABLATION_CHANNELS)
    keep = [c for c in channel_names if c not in ablate]
    y_tr = pd.Categorical(rows_train["orientation"]).codes.astype(np.int64)
    cats = pd.Categorical(rows_train["orientation"]).categories
    y_te = pd.Categorical(rows_test["orientation"], categories=cats).codes.astype(np.int64)
    floor = float(pd.Series(y_te).value_counts(normalize=True).max())

    out: List[Dict[str, Any]] = []
    for suffix, keep_t in _windows_for(times_ms, presaccadic_ms):
        sl = slice(None) if keep_t is None else keep_t
        arms = {
            "full_eeg": (x_train[:, :, sl], x_test[:, :, sl]),
            "ocular_ablated": (
                select_channels(x_train[:, :, sl], channel_names, keep=keep)[0],
                select_channels(x_test[:, :, sl], channel_names, keep=keep)[0],
            ),
            "eog_surrogate_saved": (eog_train[:, :, sl], eog_test[:, :, sl]),
        }
        for tag, (a_tr, a_te) in arms.items():
            probe = probe_factory()
            f_tr = probe.featurize(a_tr)
            f_te = probe.featurize(a_te)
            mu = f_tr.mean(axis=0, keepdims=True)
            sd = np.maximum(f_tr.std(axis=0, keepdims=True), 1e-9)
            t0 = time.time()
            clf = LogisticRegression(max_iter=300, C=0.01, random_state=seed).fit(
                (f_tr - mu) / sd, y_tr
            )
            acc = float(np.mean(clf.predict((f_te - mu) / sd) == y_te))
            above_pp = 100.0 * (acc - floor)
            out.append(
                {
                    "arm": f"{tag}{suffix}",
                    "window": suffix.lstrip("_") or "all",
                    "channels": int(a_tr.shape[1]),
                    "orientation_acc": acc,
                    "majority_floor": floor,
                    "above_floor_pp": above_pp,
                    "n_features": int(f_tr.shape[1]),
                    "n_train": int(a_tr.shape[0]),
                    "n_test": int(a_te.shape[0]),
                    "estimator": (
                        f"whole-window multinomial logistic on {f_tr.shape[1]} "
                        "features (C x time-bins), plain accuracy vs empirical floor"
                    ),
                    "status_md_above_floor_pp": STATUS_MD_MVPA_ORIENTATION["above_floor_pp"],
                    "status_md_ratio": (
                        above_pp / STATUS_MD_MVPA_ORIENTATION["above_floor_pp"]
                    ),
                    "seconds": round(time.time() - t0, 1),
                }
            )
            print(
                f"  [positive-control] {out[-1]['arm']:34s} C={a_tr.shape[1]:2d} "
                f"orientation acc={acc:.4f} floor={floor:.4f} "
                f"({out[-1]['above_floor_pp']:+.2f} pp) [{out[-1]['seconds']}s]",
                flush=True,
            )
    return out


# --------------------------------------------------------------------------- #
# battery
# --------------------------------------------------------------------------- #
def run_ocular_battery(
    *,
    subjects: Sequence[int],
    fold_index: int = 0,
    window: str = DEFAULT_WINDOW,
    emb_tag: str = "siglip2-base",
    n_video_folds: int = 5,
    seed: int = 0,
    presaccadic_ms: float = 150.0,
    n_time_bins: int = 20,
    n_pseudo_resamples: int = 10,
    pseudo_k: int = 4,
    per_subject: bool = True,
    max_train_trials: Optional[int] = None,
    positive_control: bool = True,
    trials_path: Optional[Path] = None,
    drop_channels: Optional[Sequence[str]] = None,
    gallery_scope: str = "heldout",
    primary_variant: str = PRIMARY_PREDICTION_VARIANT,
    clamp_sigmas: Optional[float] = 20.0,
    featurize_chunk: int = 4096,
) -> Dict[str, Any]:
    """Run (a), (b) and (c) on one video-disjoint fold and return everything.

    ``drop_channels`` overrides :data:`OCULAR_ABLATION_CHANNELS`.  It exists
    because the D6 list is montage-blind: on this biosemi64 layout it removes
    the lateral frontopolar pair (Fp1/Fp2) but leaves **Fpz, AFz, AF3, AF4**
    standing, and Fpz is one of the most blink-dominated sites on the head.  The
    override lets that sensitivity be measured without editing a recorded
    decision.

    ``gallery_scope``
        ``"heldout"`` (default, and the pre-registered endpoint) scores
        retrieval among the fold's held-out base videos only.  ``"all90"``
        reproduces the old behaviour of scoring against every base video; it is
        kept for diagnosis and its emitted ``gallery`` labels are prefixed
        ``leaky90_`` so those rows can never be merged against a real ``nway18``
        by :mod:`tactus.eval.report`, which joins on ``(gallery, metric,
        trial_type)``.

    ``primary_variant``
        which of :data:`~tactus.eval.probes.PREDICTION_VARIANTS` the headline
        verdict is read off.  Both are always computed and reported.

    ``clamp_sigmas``
        post-scaling clip in robust sigmas, mirroring ``data.scaling.clamp``
        (20.0).  This module gathers from the epoch memmaps directly and so does
        **not** inherit the trainer dataloader's clamp; sub-17 carries a blown
        PO4 channel that survived ICA + autoreject and that IQR-based scaling
        preserves rather than removes.  The realised clipped fraction is
        reported per arm.  ``None`` disables it -- say so if you do.

    ``featurize_chunk``
        trials per float64 featurisation block.  Numerically exact (the feature
        map is per-trial), so this only trades speed for peak memory.  It
        matters: the 180-sample window over 20 subjects peaks near the 9.9 GB
        cgroup this gate runs in, and the float64 chunk plus ``np.percentile``'s
        internal copy are the largest transient allocations in the run.
    """
    if gallery_scope not in ("heldout", "all90"):
        raise ValueError("gallery_scope must be 'heldout' or 'all90'")
    if primary_variant not in PREDICTION_VARIANTS:
        raise ValueError(
            f"unknown primary_variant {primary_variant!r}; "
            f"choose from {list(PREDICTION_VARIANTS)}"
        )
    t_start = time.time()
    spec = window_spec(window)
    times_ms = (spec.tmin + np.arange(spec.n_times) / spec.sfreq) * 1000.0

    trials = pd.read_parquet(trials_path or TRIALS_PARQUET)
    trials = trials[trials["subject_id"].isin([int(s) for s in subjects])].reset_index(drop=True)
    if trials.empty:
        raise ValueError(f"no trials for subjects {list(subjects)}")

    # video-disjoint folds; within_subject is the regime whose only holdout is
    # the base video, which is exactly what a channel-subset control needs.
    folds = make_folds(
        trials, "within_subject", n_video_folds=n_video_folds, seed=seed,
        subjects=[int(s) for s in subjects],
    )
    if not 0 <= fold_index < len(folds):
        raise IndexError(f"fold {fold_index} out of range (got {len(folds)} folds)")
    fold = folds[fold_index]
    adj = dict(fold.meta.get("adjacency") or {})
    print(f"[fold] {fold!r}", flush=True)
    print(
        f"[fold] adjacency side={adj.get('side')} dropped train={adj.get('n_dropped_train')} "
        f"test={adj.get('n_dropped_test')}",
        flush=True,
    )

    src = TrialSource(trials, [int(s) for s in subjects], window)
    train_uid = np.asarray(fold.train_uid)
    if max_train_trials is not None and train_uid.size > max_train_trials:
        rng = np.random.default_rng(seed)
        train_uid = np.sort(rng.choice(train_uid, size=int(max_train_trials), replace=False))
        print(f"[fold] subsampled train to {train_uid.size} trials", flush=True)

    x_tr, rows_tr = src.gather(train_uid, which="eeg")
    x_te, rows_te = src.gather(np.asarray(fold.test_uid), which="eeg")
    e_tr, _ = src.gather(train_uid, which="eog")
    e_te, _ = src.gather(np.asarray(fold.test_uid), which="eog")
    print(f"[data] train {x_tr.shape} test {x_te.shape} eog {e_tr.shape}", flush=True)

    # ---- gallery ------------------------------------------------------------
    # The bank (all 90) supplies TRAINING TARGETS: a train trial's target is its
    # own video's embedding.  The GALLERY is the fold's held-out base videos and
    # nothing else -- that is what "18-way" means (see module docstring and
    # tactus.baselines.linear_align._gallery_for).
    bank, bank_ids = load_video_bank(emb_tag)
    item_te = rows_te["video_id"].to_numpy(dtype=np.int64)
    heldout_ids = np.unique(item_te)
    train_video_ids = np.unique(rows_tr["video_id"].to_numpy(dtype=np.int64))
    overlap = np.intersect1d(heldout_ids, train_video_ids)
    if overlap.size:
        raise AssertionError(
            f"{overlap.size} base video(s) {overlap.tolist()[:5]} appear in both "
            "train and test -- the fold is not video-disjoint and every retrieval "
            "number below would be leakage"
        )
    fold_test_videos = np.unique(np.asarray(fold.test_videos, dtype=np.int64))
    if not np.array_equal(heldout_ids, fold_test_videos):
        # not fatal (adjacency can empty a video out of the realised test set),
        # but it must never be silent: it changes the gallery size
        print(
            f"[gallery] WARNING realised test videos ({heldout_ids.size}) differ "
            f"from fold.test_videos ({fold_test_videos.size})",
            flush=True,
        )
    if gallery_scope == "heldout":
        gallery, gallery_ids = load_gallery(emb_tag, video_ids=heldout_ids)
    else:
        gallery, gallery_ids = bank, bank_ids
    n_seen_in_gallery = int(np.intersect1d(gallery_ids, train_video_ids).size)
    print(
        f"[gallery] scope={gallery_scope} items={gallery.shape[0]} "
        f"(held-out videos in fold={heldout_ids.size}; gallery items the ridge "
        f"was trained on={n_seen_in_gallery})",
        flush=True,
    )
    if gallery_scope == "heldout" and n_seen_in_gallery:
        raise AssertionError("held-out gallery contains a training video")

    channel_names = load_channels()
    ablate = list(drop_channels) if drop_channels else list(OCULAR_ABLATION_CHANNELS)
    missing = [c for c in ablate if c not in set(channel_names)]
    if missing:
        raise KeyError(f"ocular ablation channels absent from the montage: {missing}")

    per_video = trials.drop_duplicates("video_id").set_index("video_id")
    gallery_groups = np.array(
        [str(per_video.loc[int(v), "material"]) if int(v) in per_video.index else "unknown"
         for v in gallery_ids]
    )

    y_tr = bank[rows_tr["video_id"].to_numpy(dtype=np.int64) - 1]
    subs_te = rows_te["subject_id"].to_numpy(dtype=np.int64)

    cfg = RetrievalConfig(
        n_pseudo_resamples=n_pseudo_resamples,
        pseudo_k=pseudo_k,
        directions=("eeg2vid",),
        per_subject=per_subject,
        seed=seed,
    )
    factory = lambda: RidgeAlignmentProbe(  # noqa: E731
        n_time_bins=n_time_bins, clamp_sigmas=clamp_sigmas,
        featurize_chunk=int(featurize_chunk),
    )

    # ``full_eeg`` is an arm of BOTH batteries; the probe is deterministic, so
    # the shared cache makes the second call reuse the first fit instead of
    # repeating the most expensive fit in the run.  Scoped to this battery.
    arm_cache: Dict[Any, Any] = {}

    # ---- (b)+(c) EEG vs rebuilt ocular surrogate, per window -----------------
    print("[run] ocular_control ...", flush=True)
    t0 = time.time()
    ocular = ocular_control(
        x_tr, y_tr, x_te, item_te, gallery, gallery_ids, channel_names,
        groups=gallery_groups, subject_ids_test=subs_te, times_ms=times_ms,
        presaccadic_ms=presaccadic_ms, cfg=cfg, probe_factory=factory,
        primary_variant=primary_variant, arm_cache=arm_cache,
    )
    print(f"[run] ocular_control done in {time.time() - t0:.0f}s", flush=True)

    # ---- (a)+(c) ocular-channel ablation, per window -------------------------
    print("[run] frontal_ablation_sensitivity ...", flush=True)
    t0 = time.time()
    ablation = frontal_ablation_sensitivity(
        x_tr, y_tr, x_te, item_te, gallery, gallery_ids, channel_names,
        drop_channels=ablate,
        groups=gallery_groups, subject_ids_test=subs_te, times_ms=times_ms,
        presaccadic_ms=presaccadic_ms, cfg=cfg, probe_factory=factory,
        primary_variant=primary_variant, arm_cache=arm_cache,
    )
    print(f"[run] frontal_ablation done in {time.time() - t0:.0f}s", flush=True)

    # ---- (b) the SAVED, unscaled EOG proxy ----------------------------------
    print("[run] saved EOG proxy arms ...", flush=True)
    t0 = time.time()
    saved_table, saved_diag = saved_eog_arms(
        e_tr, y_tr, e_te, item_te, gallery, gallery_ids,
        times_ms=times_ms, presaccadic_ms=presaccadic_ms, groups=gallery_groups,
        subject_ids_test=subs_te, cfg=cfg, probe_factory=factory,
    )
    print(f"[run] saved EOG done in {time.time() - t0:.0f}s", flush=True)

    table = pd.concat(
        [ocular["table"], ablation["table"], saved_table], ignore_index=True
    ).drop_duplicates(
        # ``variant`` is part of the key: without it the de-dup would keep one
        # scorer per arm and throw the other away.
        subset=["probe", "variant", "subject_id", "direction", "trial_type",
                "gallery", "metric"]
    )
    if gallery_scope != "heldout":
        # never let a 90-video row merge against a real held-out nway18 row in
        # eval/report.py, which joins on (gallery, metric, trial_type)
        table["gallery"] = "leaky90_" + table["gallery"].astype(str)
    table["gallery_scope"] = gallery_scope
    table["n_gallery_items"] = int(gallery.shape[0])

    endpoint = dict(PRIMARY_PROBE_ENDPOINT)
    if gallery_scope != "heldout":
        endpoint["gallery"] = "leaky90_" + endpoint["gallery"]
    trial_type = _resolve_trial_type(table, cfg)
    scores_by_variant: Dict[str, Dict[str, float]] = {}
    chance_by_variant: Dict[str, Dict[str, float]] = {}
    single_by_variant: Dict[str, Dict[str, Dict[str, float]]] = {}
    for v in PREDICTION_VARIANTS:
        s_v, c_v = _arm_scores(table, trial_type=trial_type, variant=v, endpoint=endpoint)
        scores_by_variant[v] = s_v
        chance_by_variant[v] = c_v
        single_by_variant[v] = arm_significance(
            table, variant=v, trial_type="single", endpoint=endpoint
        )
    scores = scores_by_variant[primary_variant]
    chance = chance_by_variant[primary_variant]
    diagnostics = {**ocular["diagnostics"], **ablation["diagnostics"], **saved_diag}

    pos = []
    if positive_control:
        print("[run] positive control (orientation) ...", flush=True)
        pos = orientation_positive_control(
            x_tr, x_te, rows_tr, rows_te, channel_names,
            times_ms=times_ms, presaccadic_ms=presaccadic_ms,
            eog_train=e_tr, eog_test=e_te, probe_factory=factory,
            drop_channels=ablate, seed=seed,
        )

    result: Dict[str, Any] = {
        "config": {
            "subjects": [int(s) for s in subjects],
            "n_subjects": len(subjects),
            "fold_index": int(fold_index),
            "fold_id": fold.fold_id,
            "n_video_folds": int(n_video_folds),
            "window": window,
            "times_ms_range": [float(times_ms[0]), float(times_ms[-1])],
            "presaccadic_ms": float(presaccadic_ms),
            "emb_tag": emb_tag,
            "embedding_dim": int(gallery.shape[1]),
            "gallery_items": int(gallery.shape[0]),
            "gallery_scope": gallery_scope,
            "gallery_video_ids": [int(v) for v in gallery_ids],
            "gallery_items_seen_in_training": n_seen_in_gallery,
            "n_heldout_videos": int(heldout_ids.size),
            "n_train_videos": int(train_video_ids.size),
            "primary_variant": primary_variant,
            "prediction_variants": list(PREDICTION_VARIANTS),
            "clamp_sigmas": (None if clamp_sigmas is None else float(clamp_sigmas)),
            "featurize_chunk": int(featurize_chunk),
            "reads_memmaps_directly": True,
            "n_train_trials": int(x_tr.shape[0]),
            "n_test_trials": int(x_te.shape[0]),
            "n_time_bins": int(n_time_bins),
            "pseudo_k": int(pseudo_k),
            "n_pseudo_resamples": int(n_pseudo_resamples),
            "seed": int(seed),
            "adjacency": adj,
            "ocular_ablation_channels": list(ablate),
            "ocular_ablation_is_d6_default": list(ablate) == list(OCULAR_ABLATION_CHANNELS),
            "channels_dropped": ablation["channels_dropped"],
            "n_channels_dropped": len(ablation["channels_dropped"]),
            "n_channels_kept": len(ablation["channels_kept"]),
        },
        "endpoint": {**endpoint, "trial_type": trial_type,
                     "subject_id": "pooled", "variant": primary_variant},
        "scores": scores,
        "chance": chance,
        "scores_by_variant": scores_by_variant,
        "chance_by_variant": chance_by_variant,
        "single_trial_by_variant": single_by_variant,
        "diagnostics": diagnostics,
        "ocular_control": {k: v for k, v in ocular.items() if k != "table"},
        "frontal_ablation": {k: v for k, v in ablation.items() if k != "table"},
        "positive_control": pos,
        "positive_control_reconciliation": {
            **STATUS_MD_MVPA_ORIENTATION,
            "this_module_estimator": (
                f"whole-window multinomial logistic on 64 x {n_time_bins} pooled "
                "bins, plain accuracy vs the empirical majority floor, "
                f"{len(subjects)} subject(s), one video-disjoint fold"
            ),
            "note": (
                "The two numbers are not the same estimator and must not be "
                "quoted as if they were: whole-window multivariate decoding "
                "integrates every timepoint, the STATUS.md figure is the peak of "
                "a single-timepoint balanced-accuracy time course. The ratio is "
                "reported per arm as 'status_md_ratio'; state it rather than "
                "letting the larger number stand unqualified."
            ),
        },
        "table": table,
        "elapsed_sec": round(time.time() - t_start, 1),
        "limitation": OCULAR_CLAIM_LIMITATION,
    }
    result["subject_level"] = subject_level_tests(
        table, variant=primary_variant, trial_type=trial_type, endpoint=endpoint,
        seed=seed,
    )
    result["g6_criterion2"] = g6_criterion2_verdict(result)
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summary_frame(result: Dict[str, Any]) -> pd.DataFrame:
    """One row per (arm, window, prediction variant) at the selected endpoint.

    Both variants are always in the frame.  The ``with_mean`` rows are the
    protocol-identical scorer and the ``nomean`` rows are the intercept-free one;
    a summary that showed only one of them is precisely how this module
    previously reported a constant-dominated scorer as a null result.
    """
    rows: List[Dict[str, Any]] = []
    scores_by_variant = result.get("scores_by_variant") or {
        result["config"].get("primary_variant", "nomean"): result["scores"]
    }
    chance_by_variant = result.get("chance_by_variant") or {
        result["config"].get("primary_variant", "nomean"): result["chance"]
    }
    for variant, scores in scores_by_variant.items():
        chance = chance_by_variant.get(variant, {})
        single = result.get("single_trial_by_variant", {}).get(variant, {})
        const_key = "constancy" if variant == "with_mean" else "constancy_nomean"
        for arm, value in sorted(scores.items()):
            base, window = arm, "all"
            for suffix in ("_presaccadic", "_sustained"):
                if arm.endswith(suffix):
                    base, window = arm[: -len(suffix)], suffix.lstrip("_")
            d = result["diagnostics"].get(arm, {})
            ch = chance.get(arm, float("nan"))
            st = single.get(arm, {})
            rows.append(
                {
                    "arm": base,
                    "window": window,
                    "variant": variant,
                    "value": value,
                    "chance": ch,
                    "above_chance": bool(value > ch),
                    "single": st.get("value", float("nan")),
                    "single_z": st.get("z_vs_chance", float("nan")),
                    "single_n": st.get("n_queries", float("nan")),
                    "constancy": d.get(const_key, float("nan")),
                    "alpha_median": d.get("alpha_median", float("nan")),
                    "alpha_at_grid_max": d.get("alpha_is_grid_max", float("nan")),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {"all": 0, "presaccadic": 1, "sustained": 2}
    v_order = {"with_mean": 0, "nomean": 1}
    return df.sort_values(
        ["window", "variant", "arm"],
        key=lambda s: (
            s.map(order) if s.name == "window"
            else s.map(v_order) if s.name == "variant" else s
        ),
    ).reset_index(drop=True)


def g6_criterion2_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    """The two clauses of G6 criterion 2, evaluated in the pre-saccadic window.

    Criterion 2 is a conjunction:

    1. the pre-saccadic (< ``presaccadic_ms``) signal **survives** frontal-channel
       ablation, and
    2. the EEG model **beats** an EOG-surrogate baseline.

    Both clauses are only meaningful if the un-ablated pre-saccadic arm clears
    chance in the first place; if it does not, the answer is ``UNDECIDED``, never
    "survives".  Numerically clearing chance is not enough either: the verdict
    additionally requires the single-trial Clopper-Pearson lower bound to sit
    above chance.

    That interval is a **screen, not a p-value**.  It is anti-conservative here
    (retrieval trials cluster within subject and within base video, so the
    effective n is far below ``n_queries``), and MISSION.md's G6 rule is explicit
    that the sanctioned null for this endpoint is the **video-level**
    permutation in :mod:`tactus.eval.permutation` -- trial-level nulls "may never
    be used to report a p value".  So the interval is used one-way: failing it is
    decisive (no signal to adjudicate), passing it is necessary but not
    sufficient, and any PASS here is labelled as resting on one fold with a
    trial-level screen plus the subject-level paired tests in
    :func:`subject_level_tests`.  A published claim still needs the video-level
    permutation null.

    Reported for the primary variant, with the other variant's numbers alongside
    so the reader can see whether the verdict is variant-dependent (it is:
    ``with_mean`` is pinned to chance by construction).
    """
    variant = result["config"]["primary_variant"]
    scores = result["scores_by_variant"][variant]
    chance = result["chance_by_variant"][variant]
    single = result.get("single_trial_by_variant", {}).get(variant, {})
    out: Dict[str, Any] = {"variant": variant, "windows": {}}
    for win, suffix in (("presaccadic", "_presaccadic"), ("all", ""),
                        ("sustained", "_sustained")):
        full = scores.get(f"full_eeg{suffix}", float("nan"))
        abl = scores.get(f"ocular_ablated{suffix}", float("nan"))
        sur = scores.get(f"ocular_surrogate{suffix}", float("nan"))
        sav = scores.get(f"eog_surrogate_saved{suffix}", float("nan"))
        ch = chance.get(f"full_eeg{suffix}", float("nan"))
        if not np.isfinite(full):
            continue
        st = single.get(f"full_eeg{suffix}", {})
        has_signal = bool(full > ch)
        ci_lo = float(st.get("ci_lo", float("nan")))
        signal_sig = bool(np.isfinite(ci_lo) and ci_lo > ch)
        surv = bool(np.isfinite(abl) and abl > ch) if has_signal else None
        beats = (
            bool(np.isfinite(sur) and full > sur and np.isfinite(sav) and full > sav)
            if has_signal else None
        )
        out["windows"][win] = {
            "full_eeg": full, "ocular_ablated": abl,
            "ocular_surrogate": sur, "eog_surrogate_saved": sav, "chance": ch,
            "full_above_chance": has_signal,
            "full_single": float(st.get("value", float("nan"))),
            "full_single_z": float(st.get("z_vs_chance", float("nan"))),
            "full_single_ci_lo": ci_lo,
            "full_single_ci_hi": float(st.get("ci_hi", float("nan"))),
            "full_single_ci_excludes_chance": signal_sig,
            "ablation_survives": surv,
            "beats_both_surrogates": beats,
            "margin_vs_rebuilt_surrogate": float(full - sur),
            "margin_vs_saved_proxy": float(full - sav),
        }
    pre = out["windows"].get("presaccadic")
    if pre is None:
        out["verdict"] = "UNDECIDED -- no pre-saccadic window in this epoch spec"
    elif not pre["full_above_chance"]:
        out["verdict"] = (
            "UNDECIDED -- the un-ablated pre-saccadic arm does not clear chance "
            f"({pre['full_eeg']:.5f} vs {pre['chance']:.5f}); with no signal "
            "present, neither 'survives ablation' nor 'beats the surrogate' can "
            "be tested in this window"
        )
    elif not pre["full_single_ci_excludes_chance"]:
        out["verdict"] = (
            "UNDECIDED -- the pre-saccadic arm is above chance at the pseudo-trial "
            f"endpoint ({pre['full_eeg']:.5f} vs {pre['chance']:.5f}) but its "
            "single-trial 95% interval still contains chance "
            f"({pre['full_single']:.5f}, CI [{pre['full_single_ci_lo']:.5f}, "
            f"{pre['full_single_ci_hi']:.5f}], z={pre['full_single_z']:+.2f}), and "
            "that interval is already anti-conservative (trials are clustered "
            "within subject and video). More subjects or more folds are needed "
            "before this window supports either clause"
        )
    elif pre["ablation_survives"] and pre["beats_both_surrogates"]:
        out["verdict"] = (
            "PASS (pre-saccadic window only; ONE fold; screened with a "
            "trial-level interval, not the video-level permutation null G6 "
            "requires) -- the pre-saccadic arm clears chance, still clears it "
            "after ocular-channel ablation, and exceeds both EOG surrogates. "
            "Read this next to the subject-level paired tests and to "
            "OCULAR_CLAIM_LIMITATION before it is quoted"
        )
    else:
        bits = []
        if not pre["ablation_survives"]:
            bits.append("the ablated arm falls to/below chance")
        if not pre["beats_both_surrogates"]:
            bits.append("it does not exceed both EOG surrogates")
        out["verdict"] = "FAIL (pre-saccadic window) -- " + "; ".join(bits)
    return out


def subject_level_tests(
    table: pd.DataFrame,
    *,
    variant: str = PRIMARY_PREDICTION_VARIANT,
    trial_type: str = "pseudo4",
    endpoint: Optional[Dict[str, str]] = None,
    contrasts: Sequence[Tuple[str, str]] = (
        ("full_eeg", "ocular_surrogate"),
        ("full_eeg", "eog_surrogate_saved"),
        ("ocular_ablated", "full_eeg"),
    ),
    windows: Sequence[str] = ("", "_presaccadic", "_sustained"),
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Paired subject-level tests at the endpoint -- the real unit of inference.

    The pooled top-1 and its binomial interval treat every retrieval trial as
    independent; they are not (trials cluster within subject and within base
    video), so that interval is anti-conservative and cannot carry a claim.  The
    pre-registered unit is the subject x fold aggregate, so each contrast here is
    a paired Wilcoxon over the ``per_subject`` rows the retrieval scorer already
    emits, plus the count of subjects favouring each arm.

    One fold means these are *within-fold* subject-level tests: they establish
    that a margin is consistent across people, not that it generalises across
    stimulus folds.  Say which one you are claiming.
    """
    from .stats import paired_wilcoxon

    if table.empty:
        return []
    ep = dict(PRIMARY_PROBE_ENDPOINT if endpoint is None else endpoint)
    arm_col = "arm" if "arm" in table.columns else "probe"
    sel = table[
        (table["subject_id"].astype(str) != "pooled")
        & (table["direction"] == ep["direction"])
        & (table["gallery"] == ep["gallery"])
        & (table["metric"] == ep["metric"])
        & (table["trial_type"] == trial_type)
    ]
    if "variant" in sel.columns:
        sel = sel[sel["variant"].astype(str) == variant]
    piv = sel.pivot_table(index="subject_id", columns=arm_col, values="value")

    out: List[Dict[str, Any]] = []
    for suffix in windows:
        for a, b in contrasts:
            ca, cb = f"{a}{suffix}", f"{b}{suffix}"
            if ca not in piv.columns or cb not in piv.columns:
                continue
            pair = piv[[ca, cb]].dropna()
            if pair.shape[0] < 3:
                continue
            res = paired_wilcoxon(
                pair[ca].to_numpy(), pair[cb].to_numpy(), labels=(ca, cb), seed=seed
            )
            out.append(
                {
                    "window": suffix.lstrip("_") or "all",
                    "variant": variant,
                    "arm": a,
                    "reference": b,
                    "mean_arm": float(pair[ca].mean()),
                    "mean_reference": float(pair[cb].mean()),
                    "mean_diff": float(res["mean_diff"]),
                    "median_diff": float(res["median_diff"]),
                    "ci_lo": float(res["ci_lo"]),
                    "ci_hi": float(res["ci_hi"]),
                    "p_value": float(res["p_value"]),
                    "n_subjects": int(res["n_pairs"]),
                    "n_favouring_arm": int((pair[ca] > pair[cb]).sum()),
                    "n_tied": int((pair[ca] == pair[cb]).sum()),
                }
            )
    return out


def mergeable_table(table: pd.DataFrame) -> pd.DataFrame:
    """The written form of the tidy table: arm names carry their scorer.

    :func:`tactus.eval.report.compare_arms` and
    :func:`~tactus.eval.report.aggregate_folds` group by
    ``DEFAULT_GROUP_COLS = (direction, trial_type, gallery, metric)`` plus the
    arm column, and **not** by ``variant``.  Handing them a two-variant table
    would average the intercept-inclusive and intercept-free scorers into one
    number per arm -- a quantity that corresponds to no analysis at all.  Since
    ``report`` is not this module's file, the defence lives here: on disk the
    ``probe`` column becomes ``"{arm}__{variant}"``, so a variant-blind groupby
    still separates them, while ``arm`` and ``variant`` remain as their own
    columns for anything that does know about them.
    """
    out = table.copy()
    if "variant" not in out.columns:
        return out
    out["arm"] = out["probe"].astype(str)
    out["probe"] = out["arm"] + "__" + out["variant"].astype(str)
    return out


def print_report(result: Dict[str, Any]) -> None:
    cfg = result["config"]
    print("\n" + "=" * 78)
    print("G6 criterion 2 -- ocular control  (RidgeAlignmentProbe, linear control)")
    print("=" * 78)
    print(
        f"subjects={cfg['n_subjects']} fold={cfg['fold_id']} window={cfg['window']} "
        f"emb={cfg['emb_tag']} (D={cfg['embedding_dim']})"
    )
    print(
        f"gallery: scope={cfg['gallery_scope']} items={cfg['gallery_items']} "
        f"(fold held out {cfg['n_heldout_videos']} of "
        f"{cfg['n_heldout_videos'] + cfg['n_train_videos']} base videos; "
        f"gallery items seen in training={cfg['gallery_items_seen_in_training']})"
    )
    print(f"train={cfg['n_train_trials']} test={cfg['n_test_trials']} trials")
    clipped = [
        (arm, d.get("clipped_fraction_train", 0.0))
        for arm, d in result["diagnostics"].items()
    ]
    worst = max((v for _, v in clipped), default=0.0)
    print(
        f"scaling: clamp={cfg['clamp_sigmas']} robust sigmas "
        f"(this module reads the memmaps directly, so it does not inherit the "
        f"trainer's clamp); max clipped fraction over arms = {worst:.2e}"
    )
    print(
        f"ablation: dropped {cfg['n_channels_dropped']}/"
        f"{cfg['n_channels_dropped'] + cfg['n_channels_kept']} channels "
        f"{cfg['channels_dropped']}"
    )
    ep = result["endpoint"]
    print(
        f"endpoint: {ep['direction']} / {ep['gallery']} / {ep['metric']} / "
        f"{ep['trial_type']} / {ep['subject_id']} / primary variant="
        f"{ep['variant']}"
    )
    print("-" * 78)
    df = summary_frame(result)
    if not df.empty:
        with pd.option_context("display.width", 220, "display.max_columns", 24):
            print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if not df.empty and df["alpha_at_grid_max"].fillna(0).max() > 0:
        hit = df.loc[df["alpha_at_grid_max"] > 0, ["arm", "window"]].drop_duplicates()
        print(
            f"[diagnostic] ridge alpha hit the top of the grid on "
            f"{len(hit)} arm-window(s) {hit.to_dict('records')} -- the fit is "
            "shrunk towards the intercept there and the effect size is a lower "
            "bound, not an estimate"
        )
    print("-" * 78)
    print("margins (full_eeg - ocular_surrogate), by prediction variant:")
    for variant, m in result["ocular_control"]["margins_by_variant"].items():
        print(f"  [{variant}] " + "  ".join(f"{k}={v:+.5f}" for k, v in m.items()))
    print("ablation retention (fraction of the above-chance margin that survives):")
    ab = result["frontal_ablation"]
    for variant in ab["retention_by_variant"]:
        ret = ab["retention_by_variant"][variant]
        ok = ab["retention_valid_by_variant"][variant]
        stable = ab.get("retention_stable_by_variant", {}).get(variant, {})
        notes = ab["retention_notes_by_variant"][variant]
        for k, v in ret.items():
            if not ok.get(k):
                print(f"  [{variant}] {k:14s} WITHHELD -- {notes.get(k, '')}")
            elif not stable.get(k, True):
                print(f"  [{variant}] {k:14s} {v:+.4f}  {notes.get(k, '')}")
            else:
                print(f"  [{variant}] {k:14s} {v:+.4f}")
    print("-" * 78)
    for variant, v in result["ocular_control"]["verdict_by_variant"].items():
        print(f"ocular_control [{variant}]: {v}")
    for variant, v in result["frontal_ablation"]["verdict_by_variant"].items():
        print(f"ablation       [{variant}]: {v}")

    subj = result.get("subject_level") or []
    if subj:
        print("-" * 78)
        print(
            "subject-level paired tests at the endpoint (the pre-registered unit "
            "of inference; within one fold):"
        )
        sdf = pd.DataFrame(subj)
        with pd.option_context("display.width", 220, "display.max_columns", 24):
            print(sdf[["window", "arm", "reference", "mean_arm", "mean_reference",
                       "mean_diff", "ci_lo", "ci_hi", "p_value", "n_subjects",
                       "n_favouring_arm"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    g6 = result.get("g6_criterion2") or g6_criterion2_verdict(result)
    print("-" * 78)
    print(f"G6 CRITERION 2 (primary variant = {g6['variant']}):")
    for win, d in g6["windows"].items():
        print(
            f"  {win:12s} full={d['full_eeg']:.5f} ablated={d['ocular_ablated']:.5f} "
            f"surrogate={d['ocular_surrogate']:.5f} saved_proxy="
            f"{d['eog_surrogate_saved']:.5f} chance={d['chance']:.5f} "
            f"| single-trial full={d['full_single']:.5f} "
            f"CI[{d['full_single_ci_lo']:.5f},{d['full_single_ci_hi']:.5f}] "
            f"z={d['full_single_z']:+.2f}"
        )
    print(f"  VERDICT: {g6['verdict']}")

    if result["positive_control"]:
        print("-" * 78)
        print("positive control (orientation decoding; NOT the gate endpoint):")
        pc = pd.DataFrame(result["positive_control"])
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(pc[["arm", "channels", "n_features", "orientation_acc",
                      "majority_floor", "above_floor_pp", "status_md_ratio"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        rec = result["positive_control_reconciliation"]
        print(
            f"  reconciliation vs STATUS.md: this module = "
            f"{rec['this_module_estimator']}; STATUS.md = {rec['estimator']} "
            f"(+{rec['above_floor_pp']} pp, n={rec['n_subjects']}, "
            f"cv={rec['cv']}). {rec['note']}"
        )
    print("-" * 78)
    print("OCULAR CLAIM LIMITATION (must appear in the manuscript body):")
    print(OCULAR_CLAIM_LIMITATION)
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tactus.eval.run_ocular",
        description="Gate G6 criterion 2: ocular ablation + EOG-surrogate baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subjects", default="1-8", help="e.g. '1-8' or '1,3,5'")
    p.add_argument("--fold", type=int, default=0, help="which video fold to run")
    p.add_argument("--n-video-folds", type=int, default=5)
    p.add_argument("--window", default=DEFAULT_WINDOW, help="w0600 or wm100_800")
    p.add_argument("--emb", default="siglip2-base", help="video embedding bank tag")
    p.add_argument("--presaccadic-ms", type=float, default=150.0)
    p.add_argument("--n-time-bins", type=int, default=20)
    p.add_argument("--pseudo-k", type=int, default=4)
    p.add_argument("--n-pseudo-resamples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-trials", type=int, default=None)
    p.add_argument("--no-per-subject", action="store_true")
    p.add_argument(
        "--drop-channels",
        default=None,
        help="comma-separated ablation list overriding the D6 default "
        "(Fp1,Fp2,AF7,AF8,F7,F8,F5,F6). Note the D6 list leaves Fpz/AFz/AF3/AF4 "
        "standing on this montage; 'Fp1,Fp2,Fpz,AF7,AF8,AFz,F7,F8,F5,F6' is the "
        "blink-complete variant.",
    )
    p.add_argument("--no-positive-control", action="store_true")
    p.add_argument(
        "--gallery-scope",
        default="heldout",
        choices=["heldout", "all90"],
        help="'heldout' (default, the pre-registered 18-way endpoint) retrieves "
        "among the fold's held-out base videos only. 'all90' scores against every "
        "base video -- a DIFFERENT, easier endpoint in which most nway18 "
        "distractors were in the training set; its gallery labels are prefixed "
        "'leaky90_' so they cannot merge with real nway18 rows.",
    )
    p.add_argument(
        "--primary-variant",
        default=PRIMARY_PREDICTION_VARIANT,
        choices=list(PREDICTION_VARIANTS),
        help="which prediction variant the headline verdict is read off; both "
        "are always computed and reported. 'with_mean' is the protocol-identical "
        "scorer but is pinned to chance by the query-independent intercept on "
        "this anisotropic gallery.",
    )
    p.add_argument(
        "--clamp-sigmas",
        type=float,
        default=20.0,
        help="post-scaling clip in robust sigmas, mirroring data.scaling.clamp. "
        "This module reads the epoch memmaps directly and does NOT inherit the "
        "trainer's clamp; sub-17 carries a blown PO4 channel that IQR scaling "
        "preserves. Pass a non-finite value (inf) to disable.",
    )
    p.add_argument(
        "--featurize-chunk",
        type=int,
        default=4096,
        help="trials per float64 featurisation block. Numerically exact; lower "
        "it to fit a tighter memory cgroup (the 180-sample window over 20 "
        "subjects peaks near 10 GB at the default).",
    )
    p.add_argument("--trials", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="directory for csv + json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_ocular_battery(
        subjects=parse_subjects(args.subjects),
        fold_index=args.fold,
        window=args.window,
        emb_tag=args.emb,
        n_video_folds=args.n_video_folds,
        seed=args.seed,
        presaccadic_ms=args.presaccadic_ms,
        n_time_bins=args.n_time_bins,
        n_pseudo_resamples=args.n_pseudo_resamples,
        pseudo_k=args.pseudo_k,
        per_subject=not args.no_per_subject,
        max_train_trials=args.max_train_trials,
        positive_control=not args.no_positive_control,
        trials_path=args.trials,
        gallery_scope=args.gallery_scope,
        primary_variant=args.primary_variant,
        clamp_sigmas=(
            None if not np.isfinite(args.clamp_sigmas) else float(args.clamp_sigmas)
        ),
        featurize_chunk=args.featurize_chunk,
        drop_channels=(
            [c.strip() for c in args.drop_channels.split(",") if c.strip()]
            if args.drop_channels
            else None
        ),
    )
    print_report(result)

    if args.out is not None:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        tag = (
            f"sub{len(result['config']['subjects'])}_fold{args.fold}_{args.window}"
            f"_{args.emb}_{args.gallery_scope}"
        )
        mergeable_table(result["table"]).to_csv(out / f"ocular_table_{tag}.csv", index=False)
        summary_frame(result).to_csv(out / f"ocular_summary_{tag}.csv", index=False)
        payload = {k: v for k, v in result.items() if k != "table"}
        (out / f"ocular_{tag}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"[written] {out}/ocular_{tag}.json (+ _table_/_summary_ csv)")

    # exit 0 = ran; the verdict is a scientific result, not a process failure
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
