"""Confound battery for TACTUS.

Five probes, each written so that its *failure mode* is reported next to its
result rather than buried:

(a) :func:`subject_identity_probe` -- how much subject identity survives in the
    learned embedding, reported **jointly** with the alignment score on the same
    embedding.  Identity accuracy alone is a metric you can win by destroying
    all information, so the API refuses to return it without the alignment
    number it must be traded against.

(b) :func:`trial_index_control` -- can the labels be predicted from trial index,
    sequence id and presentation number alone?  If yes, the "decoding" is a
    clock.  Run both within-subject (hold out whole sequences) and
    across-subject (hold out subjects); the second is the one that catches a
    presentation order shared across the 80 participants.

(c) :func:`ocular_control` -- the same alignment pipeline trained on HEOG/VEOG
    surrogates derived from frontal channels (no EOG channels exist in
    ds005662).  The EEG model must beat it.  Beating it does **not** license an
    unqualified "not ocular" claim outside the pre-saccadic window; see
    :data:`OCULAR_CLAIM_LIMITATION`, which is emitted with every result.

(d) :func:`frontal_ablation_sensitivity` -- rerun with the eight ocular channels
    removed (:data:`OCULAR_ABLATION_CHANNELS`, decision D6), optionally split
    into the pre-saccadic and sustained windows.

Both (c) and (d) report ``informative``: when the un-ablated ``full_eeg`` arm is
itself at chance, or the linear probe collapsed onto a constant prediction, they
return ``UNINFORMATIVE`` rather than a comparison.  A control with no signal to
ablate cannot support "survives ablation" *or* "does not beat the surrogate".
(d) additionally withholds ``retention`` in that case: below chance the ratio
``(ablated - chance) / (full - chance)`` is two negatives and prints as "+1.00,
the signal survives" exactly when there is no signal.

Two things about (c)/(d) that decide whether their numbers mean anything:

* **the gallery must be the endpoint's gallery.**  ``nway18`` is the
  pre-registered retrieval among a fold's 18 *held-out* base videos.  Handing
  these functions all 90 videos still emits rows labelled ``gallery=nway18`` --
  an analytic 17-distractor draw in which most distractors were in the training
  set -- which then merges against the deep model's numbers as if comparable.
* **both prediction variants are scored** (:data:`PREDICTION_VARIANTS`).  The
  fitted intercept is a query-independent constant that, on an anisotropic
  gallery, dominates the EEG-dependent term and pins the score to chance; a
  battery that scores only ``with_mean`` reports "no signal" for every possible
  input.

(e) :func:`compute_lowlevel_features` / :func:`lowlevel_control_analysis` --
    optical-flow energy, luminance and contrast per clip (OpenCV), used both as
    an alternative "gallery" (how far can low-level features alone get?) and as
    nuisance regressors partialled out of the video embedding.

The linear :class:`RidgeAlignmentProbe` exists so that (c) and (d) can be run
without retraining the deep model 40 times: it is a fast, transparent
EEG -> video-embedding regressor whose retrieval numbers are directly
comparable across channel subsets.  It is a *control*, not a baseline claim.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .permutation import permutation_pvalue
from .retrieval import (
    RetrievalConfig,
    evaluate_retrieval,
    l2_normalize,
    to_numpy,
)

__all__ = [
    "OCULAR_CLAIM_LIMITATION",
    "DEFAULT_FRONTAL_PREFIXES",
    "OCULAR_ABLATION_CHANNELS",
    "PRIMARY_PROBE_ENDPOINT",
    "PREDICTION_VARIANTS",
    "PRIMARY_PREDICTION_VARIANT",
    "DEGENERATE_CONSTANCY",
    "RidgeAlignmentProbe",
    "arm_significance",
    "ocular_surrogates",
    "select_channels",
    "subject_identity_probe",
    "identity_alignment_tradeoff",
    "trial_index_control",
    "ocular_control",
    "frontal_ablation_sensitivity",
    "compute_lowlevel_features",
    "build_lowlevel_matrix",
    "residualize",
    "lowlevel_control_analysis",
    "run_confound_battery",
]

ArrayLike = Any

#: Named limitation that must travel with every ocular-control result.
OCULAR_CLAIM_LIMITATION = (
    "ds005662 contains zero EOG channels. HEOG/VEOG here are frontal-channel "
    "surrogates (F7-F8; mean(Fp1,Fp2) - scalp mean), not measured eye movements. "
    "Two consequences, both directional: (i) the surrogate carries genuine "
    "frontal neural signal, so beating it is anti-conservative for early "
    "components; (ii) a saccadic spike potential can volume-conduct to posterior "
    "sites, so beating a frontal surrogate does not exclude an ocular "
    "contribution at posterior sensors. Latency claims of the form 'not ocular' "
    "are defensible only inside the pre-saccadic window (< ~150 ms). Any claim "
    "about the sustained window (including the ~300 ms valence peak) must carry "
    "this limitation in the text, not in a footnote."
)

#: Channel-name prefixes treated as frontal for the ablation / surrogate probes.
#: **Do not use these for the ocular sensitivity analysis** -- see
#: :data:`OCULAR_ABLATION_CHANNELS`.  Kept only for callers that genuinely want
#: a whole-frontal-lobe ablation and say so.
DEFAULT_FRONTAL_PREFIXES: Tuple[str, ...] = ("Fp", "AF", "F")

#: The two ways one fitted ridge prediction can be scored, both always reported.
#:
#: ``with_mean``
#:     the raw ridge prediction ``x @ W + b``.  Protocol-identical -- the same
#:     scorer the contrastive model is graded with.
#: ``nomean``
#:     the same prediction with the fitted intercept ``b`` (the training-mean
#:     embedding) removed, i.e. the EEG-*dependent* part only.
#:
#: ``b`` is a constant shared by every query, so under cosine retrieval it adds
#: a query-independent bias ``b . g_k`` to gallery item ``k``.  On an
#: anisotropic gallery (SigLIP2: mean pairwise cosine 0.88) that bias dominates
#: the EEG-dependent term by roughly an order of magnitude and pins the score to
#: chance regardless of what the EEG contains -- the identical failure documented
#: in :meth:`tactus.baselines.linear_align.RidgeAlpha.predict`.  Scoring only
#: ``with_mean`` therefore cannot distinguish "no signal" from "signal buried
#: under a constant", which is exactly the distinction an ocular control exists
#: to make.  Neither variant is silently preferred: both are computed, both are
#: emitted (column ``variant``), and every verdict names the one it used.
PREDICTION_VARIANTS: Tuple[str, ...] = ("with_mean", "nomean")

#: The variant the ocular/ablation verdicts are read off by default.
#:
#: ``nomean``, because these two probes are *sensitivity* analyses: the question
#: is whether a signal changes when channels or a time window are removed, and a
#: scorer that is pinned to chance by construction answers "no change" for every
#: possible input.  The ``with_mean`` numbers are still reported next to it and
#: must be quoted whenever the comparison to the deep model's own scorer is the
#: point.
PRIMARY_PREDICTION_VARIANT: str = "nomean"

#: Explicit channel list for the ocular ablation (decision D6).
#:
#: The prefix form above is a trap here: matching ``"F"`` against the start of a
#: biosemi64 label also removes FC1-FC6, FT7/FT8 and Fz, i.e. 20+ channels of
#: central-frontal cortex.  An "is this eye movements?" sensitivity analysis that
#: silently ablates the entire frontal lobe cannot answer the question it was
#: asked -- a drop in accuracy would be uninterpretable.  These eight are the
#: sites where corneo-retinal dipole and lid artifact actually dominate.
OCULAR_ABLATION_CHANNELS: Tuple[str, ...] = (
    "Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "F5", "F6",
)


# --------------------------------------------------------------------------- #
# channel utilities
# --------------------------------------------------------------------------- #
def _channel_index(channel_names: Sequence[str], name: str) -> int:
    lookup = {c.strip().lower(): i for i, c in enumerate(channel_names)}
    key = name.strip().lower()
    if key not in lookup:
        raise KeyError(
            f"channel {name!r} not found; available (first 10): "
            f"{list(channel_names)[:10]}"
        )
    return lookup[key]


def select_channels(
    x: ArrayLike,
    channel_names: Sequence[str],
    *,
    keep: Optional[Sequence[str]] = None,
    drop_prefixes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Subset (n_trials, C, T) epochs by channel name or by name prefix.

    ``drop_prefixes`` is matched case-insensitively against the *start* of the
    channel label, with the deliberate consequence that ``"F"`` removes F1-F8,
    FC*, FT* and Fp*/AF* when combined with the defaults.  Pass an explicit
    ``keep`` list when a surgical subset is wanted.
    """
    x_arr = np.asarray(to_numpy(x))
    names = [str(c) for c in channel_names]
    if keep is not None:
        idx = [_channel_index(names, c) for c in keep]
    elif drop_prefixes is not None:
        low = tuple(p.lower() for p in drop_prefixes)
        idx = [i for i, c in enumerate(names) if not c.strip().lower().startswith(low)]
    else:
        idx = list(range(len(names)))
    if not idx:
        raise ValueError("channel selection removed every channel")
    return x_arr[:, idx, :], [names[i] for i in idx]


def ocular_surrogates(
    x: ArrayLike,
    channel_names: Sequence[str],
    *,
    heog_pair: Tuple[str, str] = ("F7", "F8"),
    veog_frontal: Sequence[str] = ("Fp1", "Fp2"),
    veog_reference: str = "scalp_mean",
) -> Tuple[np.ndarray, List[str]]:
    """Construct HEOG/VEOG surrogates from frontal channels (BLUEPRINT v2 §6.3).

    Parameters
    ----------
    x : (n_trials, C, T).
    veog_reference : ``"scalp_mean"`` (Fp mean minus the mean of all channels,
        the usual bipolar-free approximation) or a channel name.

    Returns
    -------
    (n_trials, 2, T) array with channels ``["HEOG_surrogate", "VEOG_surrogate"]``.
    """
    x_arr = np.asarray(to_numpy(x), dtype=np.float32)
    names = [str(c) for c in channel_names]
    i_left = _channel_index(names, heog_pair[0])
    i_right = _channel_index(names, heog_pair[1])
    heog = x_arr[:, i_left, :] - x_arr[:, i_right, :]

    fp_idx = [_channel_index(names, c) for c in veog_frontal]
    fp_mean = x_arr[:, fp_idx, :].mean(axis=1)
    if veog_reference == "scalp_mean":
        ref = x_arr.mean(axis=1)
    else:
        ref = x_arr[:, _channel_index(names, veog_reference), :]
    veog = fp_mean - ref

    return np.stack([heog, veog], axis=1), ["HEOG_surrogate", "VEOG_surrogate"]


# --------------------------------------------------------------------------- #
# a fast, transparent linear alignment probe
# --------------------------------------------------------------------------- #
@dataclass
class RidgeAlignmentProbe:
    """Ridge regression from EEG epochs to (frozen) video embeddings.

    Feature map: robust per-channel scaling, average-pool time into
    ``n_time_bins`` bins, flatten to ``C * n_time_bins``, standardise.
    Target: the L2-normalised video embedding of the trial's condition.
    Prediction is re-normalised, so retrieval uses cosine similarity exactly as
    the contrastive model does.

    This is intentionally weak and intentionally fast: its job is to make
    channel-subset comparisons (full EEG vs ocular surrogate vs frontal-ablated)
    on identical footing, not to be a headline baseline.
    """

    n_time_bins: int = 20
    alphas: Tuple[float, ...] = tuple(np.logspace(1, 6, 12))
    robust_scale: bool = True
    fit_intercept: bool = True
    #: post-scaling clip in robust sigmas, mirroring ``data.scaling.clamp`` in
    #: ``configs/default.yaml``.  This is not cosmetic: the frozen epoch memmaps
    #: still contain at least one blown channel (sub-17 PO4, channel sd 273.7 vs
    #: a 3.31 median, max |x| = 33298 robust sigmas) which survived ICA and
    #: autoreject, and per-channel robust scaling normalises by the IQR, which is
    #: *robust to* -- and therefore preserves -- an isolated spike.  The trainer's
    #: dataloader clamps and is protected; anything reading the memmaps directly,
    #: including this probe, is not.  Set to ``None``/``inf`` to disable, and say
    #: so in the report if you do.  The clamp is applied to the robust-sigma
    #: values before time-pooling, exactly where the dataloader applies it.
    clamp_sigmas: Optional[float] = 20.0
    #: trials per featurisation chunk.  The feature map is strictly per-trial,
    #: so chunking is numerically exact; it exists because the float64 cast of a
    #: full 80-subject train split is ~11 GB and ``np.percentile`` allocates a
    #: second copy of the same size, which OOMs the node the gate runs on.
    featurize_chunk: int = 4096
    _mu: Optional[np.ndarray] = field(default=None, repr=False)
    _sd: Optional[np.ndarray] = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    #: populated by :meth:`predict_variants`; see :meth:`_diagnose`.  Flat dict
    #: for the ``with_mean`` variant, plus ``*_nomean`` copies of every key for
    #: the intercept-free variant and ``intercept_dominance``.
    diagnostics_: Dict[str, float] = field(default_factory=dict, repr=False)
    #: ``{variant: diagnostics}``, also populated by :meth:`predict_variants`.
    diagnostics_by_variant_: Dict[str, Dict[str, float]] = field(
        default_factory=dict, repr=False
    )
    #: fraction of scaled samples hit by ``clamp_sigmas`` on the last
    #: :meth:`featurize` call; recorded per fit/predict so a clamp that is doing
    #: real work (a blown channel) is visible instead of implicit.
    clipped_fraction_: float = field(default=0.0, repr=False)
    _n_clipped: int = field(default=0, repr=False)
    _n_samples: int = field(default=0, repr=False)

    #: IQR -> robust sigma, so ``clamp_sigmas`` is in the same units as
    #: ``data.scaling.clamp`` (which divides the IQR by this constant).
    #: ``ClassVar`` so the dataclass does not turn it into a per-instance field.
    IQR_TO_SIGMA: ClassVar[float] = 1.349

    # ---- features ---------------------------------------------------------
    def _featurize_block(self, x_arr: np.ndarray) -> np.ndarray:
        n, c, t = x_arr.shape
        if self.robust_scale:
            med = np.median(x_arr, axis=2, keepdims=True)
            iqr = np.subtract(*np.percentile(x_arr, [75, 25], axis=2))[:, :, None]
            # in robust sigmas; the constant factor is undone by the per-feature
            # standardisation in ``fit``, so it changes nothing except the units
            # the clamp below is expressed in
            x_arr = (x_arr - med) / np.maximum(iqr, 1e-9) * self.IQR_TO_SIGMA
            lim = self.clamp_sigmas
            if lim is not None and np.isfinite(lim):
                over = int(np.count_nonzero(np.abs(x_arr) > lim))
                self._n_clipped += over
                self._n_samples += x_arr.size
                if over:
                    np.clip(x_arr, -lim, lim, out=x_arr)
        bins = min(self.n_time_bins, t)
        edges = np.linspace(0, t, bins + 1).astype(int)
        pooled = np.stack(
            [x_arr[:, :, edges[i] : edges[i + 1]].mean(axis=2) for i in range(bins)],
            axis=2,
        )
        return pooled.reshape(n, c * bins)

    def featurize(self, x: ArrayLike) -> np.ndarray:
        """(n_trials, C, T) -> (n_trials, C * n_time_bins).

        Processed in chunks of ``featurize_chunk`` trials.  Every operation in
        the feature map is independent across trials, so the result is identical
        to featurising the whole array at once; only the peak memory differs.
        """
        x_any = to_numpy(x)
        if x_any.ndim != 3:
            raise ValueError("expected (n_trials, C, T)")
        n, c, t = x_any.shape
        bins = min(self.n_time_bins, t)
        step = max(1, int(self.featurize_chunk))
        out = np.empty((n, c * bins), dtype=np.float64)
        self._n_clipped = 0
        self._n_samples = 0
        for a in range(0, n, step):
            b = min(a + step, n)
            out[a:b] = self._featurize_block(np.asarray(x_any[a:b], dtype=np.float64))
        self.clipped_fraction_ = (
            self._n_clipped / self._n_samples if self._n_samples else 0.0
        )
        return out

    # ---- fit / predict ----------------------------------------------------
    def fit(self, x: ArrayLike, y: ArrayLike) -> "RidgeAlignmentProbe":
        from sklearn.linear_model import Ridge, RidgeCV

        feats = self.featurize(x)
        self.clipped_fraction_train_ = float(self.clipped_fraction_)
        self._mu = feats.mean(axis=0, keepdims=True)
        self._sd = np.maximum(feats.std(axis=0, keepdims=True), 1e-9)
        feats = (feats - self._mu) / self._sd
        targets = l2_normalize(np.asarray(to_numpy(y), dtype=np.float64))
        try:
            self._model = RidgeCV(
                alphas=np.asarray(self.alphas), fit_intercept=self.fit_intercept
            ).fit(feats, targets)
        except Exception as exc:  # pragma: no cover - RidgeCV multi-output edge cases
            warnings.warn(f"RidgeCV failed ({exc}); falling back to a fixed alpha")
            self._model = Ridge(
                alpha=float(np.median(self.alphas)), fit_intercept=self.fit_intercept
            ).fit(feats, targets)
        return self

    def _raw_predict(self, x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """``(raw_prediction (n, D), intercept (D,))`` before normalisation."""
        if self._model is None:
            raise RuntimeError("probe is not fitted")
        feats = (self.featurize(x) - self._mu) / self._sd
        raw = np.atleast_2d(np.asarray(self._model.predict(feats), dtype=np.float64))
        b = np.asarray(getattr(self._model, "intercept_", 0.0), dtype=np.float64).ravel()
        if b.size == 1:
            b = np.full(raw.shape[1], float(b[0]))
        if b.shape[0] != raw.shape[1]:
            raise ValueError(
                f"intercept has {b.shape[0]} entries for a {raw.shape[1]}-d target"
            )
        return raw, b

    def predict_variants(self, x: ArrayLike) -> Dict[str, np.ndarray]:
        """Both scorable forms of one prediction; see :data:`PREDICTION_VARIANTS`.

        ``with_mean`` is the raw ridge output, ``nomean`` is the same output with
        the fitted intercept (the training-mean embedding) subtracted *before*
        L2-normalisation.  Removing it after normalisation would not be the same
        operation, and would not remove the query-independent bias.

        Populates :attr:`diagnostics_by_variant_` and :attr:`diagnostics_`.
        """
        raw, b = self._raw_predict(x)
        centred = raw - b[None, :]
        out = {"with_mean": l2_normalize(raw), "nomean": l2_normalize(centred)}
        self.diagnostics_by_variant_ = {k: self._diagnose(v) for k, v in out.items()}
        # how badly does the query-independent constant swamp the EEG-dependent
        # part?  >1 means the intercept is longer than the typical signal vector.
        eeg_norm = float(np.mean(np.linalg.norm(centred, axis=1)))
        dom = float(np.linalg.norm(b) / max(eeg_norm, 1e-12))
        flat = dict(self.diagnostics_by_variant_["with_mean"])
        for key, val in self.diagnostics_by_variant_["nomean"].items():
            flat[f"{key}_nomean"] = val
        flat["intercept_dominance"] = dom
        flat["intercept_is_zero"] = float(np.linalg.norm(b) < 1e-12)
        flat["clipped_fraction_train"] = float(getattr(self, "clipped_fraction_train_", 0.0))
        flat["clipped_fraction_test"] = float(self.clipped_fraction_)
        flat["clamp_sigmas"] = float(
            self.clamp_sigmas if self.clamp_sigmas is not None else np.inf
        )
        self.diagnostics_ = flat
        return out

    def predict(self, x: ArrayLike, *, variant: str = "with_mean") -> np.ndarray:
        """One prediction variant.  ``variant`` must be in :data:`PREDICTION_VARIANTS`.

        The default is ``with_mean`` so that this stays the protocol-identical
        prediction for existing callers; pass ``variant="nomean"`` for the
        intercept-free (EEG-dependent) part, or use :meth:`predict_variants` to
        get both from a single featurisation.
        """
        if variant not in PREDICTION_VARIANTS:
            raise ValueError(
                f"unknown prediction variant {variant!r}; "
                f"choose from {list(PREDICTION_VARIANTS)}"
            )
        return self.predict_variants(x)[variant]

    # ---- degeneracy diagnostics -------------------------------------------
    def _diagnose(self, z: np.ndarray) -> Dict[str, float]:
        """Is this probe actually predicting anything trial-specific?

        A ridge fitted to a 768-d, mean-dominated target with weak features
        minimises MSE by collapsing onto the intercept.  The predictions are
        then a *constant* unit vector, and cosine retrieval against the gallery
        returns a fixed ordering that has nothing to do with the EEG -- but it
        is not chance either, so a naive read of the accuracy invents a result.
        ``constancy`` is the norm of the mean predicted unit vector: 1.0 means
        every trial got the same embedding, 0 means maximal spread.
        """
        if z.ndim != 2 or z.shape[0] < 2:
            return {}
        constancy = float(np.linalg.norm(z.mean(axis=0)))
        alpha = getattr(self._model, "alpha_", None)
        out = {
            "constancy": constancy,
            "dispersion": float(np.sqrt(max(0.0, 1.0 - constancy ** 2))),
            "n_predicted": float(z.shape[0]),
        }
        if alpha is not None:
            out["alpha_median"] = float(np.median(np.atleast_1d(alpha)))
            out["alpha_is_grid_max"] = float(
                np.median(np.atleast_1d(alpha)) >= max(self.alphas) * (1 - 1e-9)
            )
        return out

    # ---- end-to-end -------------------------------------------------------
    def fit_evaluate(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_test: ArrayLike,
        test_item_id: ArrayLike,
        gallery_emb: ArrayLike,
        gallery_item_ids: ArrayLike,
        *,
        groups: Optional[ArrayLike] = None,
        subject_ids_test: Optional[ArrayLike] = None,
        cfg: Optional[RetrievalConfig] = None,
        tag: str = "ridge",
        variants: Sequence[str] = PREDICTION_VARIANTS,
    ) -> pd.DataFrame:
        """Fit on train, predict test, and return the tidy retrieval frame.

        The frame carries a ``variant`` column (see :data:`PREDICTION_VARIANTS`)
        with one block of rows per requested prediction variant, from a single
        fit and a single featurisation of the test set.  Downstream selection
        **must** name the variant: two variants of the same arm share the
        ``(probe, subject_id, direction, gallery, metric, trial_type)`` key, so a
        selector that ignores ``variant`` silently keeps whichever sorted last.
        """
        bad = [v for v in variants if v not in PREDICTION_VARIANTS]
        if bad:
            raise ValueError(
                f"unknown prediction variant(s) {bad}; "
                f"choose from {list(PREDICTION_VARIANTS)}"
            )
        self.fit(x_train, y_train)
        preds = self.predict_variants(x_test)
        cfg = cfg or RetrievalConfig(n_pseudo_resamples=10, directions=("eeg2vid",))
        frames = [
            evaluate_retrieval(
                preds[v], test_item_id, gallery_emb, gallery_item_ids,
                groups=groups, subject_ids=subject_ids_test, cfg=cfg,
                extra_fields={"probe": tag, "variant": v},
            )
            for v in variants
        ]
        return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# (a) subject identity probe -- always joint with alignment
# --------------------------------------------------------------------------- #
def subject_identity_probe(
    z: ArrayLike,
    subject_ids: ArrayLike,
    alignment_score: float,
    *,
    alignment_name: str = "eeg2vid_nway18_top1",
    alignment_chance: float = 1.0 / 18,
    video_ids: Optional[ArrayLike] = None,
    n_splits: int = 5,
    seed: int = 0,
    max_iter: int = 2000,
    model: str = "logistic",
) -> Dict[str, Any]:
    """Decode subject identity from learned embeddings, reported with alignment.

    ``alignment_score`` is required, not optional: an invariance number without
    the retained-performance number is unfalsifiable (a constant embedding wins
    the identity metric outright).  The returned dict always contains both plus
    the trade-off summary.

    The cross-validation splits by **video** whenever ``video_ids`` is supplied,
    so identity cannot be read off stimulus-specific responses shared within a
    training video.

    Returns
    -------
    dict with ``identity_top1``, ``identity_chance``, ``identity_balanced_acc``,
    ``identity_above_chance``, ``alignment_score``, ``alignment_chance``,
    ``alignment_retention``, ``invariance_index`` and ``n_subjects``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    z_arr = np.asarray(to_numpy(z), dtype=np.float64)
    if z_arr.ndim > 2:
        z_arr = z_arr.reshape(z_arr.shape[0], -1)
    subs = np.asarray(to_numpy(subject_ids)).ravel()
    n_sub = int(np.unique(subs).size)
    if n_sub < 2:
        raise ValueError("subject identity probe needs >= 2 subjects")

    if video_ids is not None:
        groups = np.asarray(to_numpy(video_ids)).ravel()
        splitter = GroupKFold(n_splits=min(n_splits, int(np.unique(groups).size)))
        split_iter = splitter.split(z_arr, subs, groups)
        split_kind = "video-disjoint"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(z_arr, subs)
        split_kind = "stratified-trial (WEAK: no video disjointness)"
        warnings.warn(
            "subject_identity_probe without video_ids uses trial-level splits; "
            "identity accuracy will be optimistic.",
            RuntimeWarning,
        )

    if model != "logistic":
        raise ValueError("only the linear 'logistic' probe is supported")

    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    for tr, te in split_iter:
        scaler = StandardScaler().fit(z_arr[tr])
        clf = LogisticRegression(max_iter=max_iter, C=1.0).fit(
            scaler.transform(z_arr[tr]), subs[tr]
        )
        y_true.append(subs[te])
        y_pred.append(clf.predict(scaler.transform(z_arr[te])))
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)

    acc = float(np.mean(yt == yp))
    bal = float(balanced_accuracy_score(yt, yp))
    chance = 1.0 / n_sub
    retention = float(
        (alignment_score - alignment_chance) / max(1e-12, 1.0 - alignment_chance)
    )
    excess_identity = float(
        (acc - chance) / max(1e-12, 1.0 - chance)
    )
    return {
        "identity_top1": acc,
        "identity_balanced_acc": bal,
        "identity_chance": chance,
        "identity_above_chance": float(acc - chance),
        "identity_excess_normalized": excess_identity,
        "alignment_name": alignment_name,
        "alignment_score": float(alignment_score),
        "alignment_chance": float(alignment_chance),
        "alignment_retention": retention,
        # a single number is only ever a summary of the pair; both are above
        "invariance_index": float(retention - excess_identity),
        "n_subjects": n_sub,
        "n_trials": int(z_arr.shape[0]),
        "split_kind": split_kind,
        "caveat": (
            "Identity accuracy must be read against alignment_retention on the "
            "same embedding. Lower identity accuracy is only meaningful at "
            "matched or better alignment."
        ),
    }


def identity_alignment_tradeoff(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Assemble an invariance/information trade-off curve.

    ``records`` are outputs of :func:`subject_identity_probe`, typically one per
    subject-adversarial weight (lambda_1).  Returns the frame sorted by
    alignment with a ``pareto`` flag marking points that are not dominated
    (higher alignment AND lower identity) by any other point -- the only points
    that support an invariance claim.
    """
    df = pd.DataFrame(list(records))
    if df.empty:
        return df
    df = df.sort_values("alignment_score", ascending=False).reset_index(drop=True)
    pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if (
                df.loc[j, "alignment_score"] >= df.loc[i, "alignment_score"]
                and df.loc[j, "identity_top1"] <= df.loc[i, "identity_top1"]
                and (
                    df.loc[j, "alignment_score"] > df.loc[i, "alignment_score"]
                    or df.loc[j, "identity_top1"] < df.loc[i, "identity_top1"]
                )
            ):
                pareto[i] = False
                break
    df["pareto"] = pareto
    return df


# --------------------------------------------------------------------------- #
# (b) trial-index / time control decoder
# --------------------------------------------------------------------------- #
def _block_permute_cells(
    cells: np.ndarray,
    videos: np.ndarray,
    subjects: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, bool]:
    """Break the timing -> label association while preserving trial clusters.

    Within each subject, the *set of timing cells* belonging to one base video
    is reassigned to another base video (position-matched by within-video
    ordinal).  Relabelling the videos instead -- the shuffle used everywhere
    else in this package -- is degenerate here: a cell -> label lookup remains
    perfect under any coherent relabelling, so that null would report p = 1 for
    a decoder at 100% accuracy.

    Returns ``(permuted_cells, exact)``; ``exact`` is False when the design is
    ragged (unequal trial counts per video within a subject) and the fallback
    -- a within-subject trial-level shuffle of the cells -- had to be used.
    """
    out = np.empty_like(cells)
    exact = True
    for s in np.unique(subjects):
        s_idx = np.flatnonzero(subjects == s)
        vids = videos[s_idx]
        uniq = np.unique(vids)
        blocks = [s_idx[vids == v] for v in uniq]
        sizes = {b.size for b in blocks}
        if len(sizes) != 1:
            exact = False
            out[s_idx] = cells[rng.permutation(s_idx)]
            continue
        perm = rng.permutation(len(blocks))
        for dst, src in enumerate(perm):
            out[blocks[dst]] = cells[blocks[src]]
    return out, exact


def _cell_mode_cv_accuracy(
    cells: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int
) -> float:
    """Leave-group-out accuracy of a modal-label lookup keyed on ``cells``.

    Deliberately not a fitted model.  For this confound the maximally powerful
    decoder is the lookup table "what label usually appears in this timing
    cell", and it costs O(n) instead of the hours a 90-class gradient-boosting
    model would take over 230k trials x 5 folds x every permutation.
    """
    uniq_groups = np.unique(groups)
    k = int(min(n_splits, uniq_groups.size))
    if k < 2:
        return float("nan")
    fold_of_group = {g: i % k for i, g in enumerate(uniq_groups)}
    fold = np.array([fold_of_group[g] for g in groups])

    accs: List[float] = []
    for f in range(k):
        tr = fold != f
        te = ~tr
        if not te.any() or not tr.any():
            continue
        global_mode = np.bincount(y[tr]).argmax()
        lookup: Dict[Any, int] = {}
        order = np.lexsort((y[tr], cells[tr]))
        c_sorted, y_sorted = cells[tr][order], y[tr][order]
        bounds = np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1]])
        for s, e in zip(bounds, np.r_[bounds[1:], c_sorted.size]):
            vals, counts = np.unique(y_sorted[s:e], return_counts=True)
            lookup[c_sorted[s]] = int(vals[counts.argmax()])
        pred = np.array([lookup.get(c, global_mode) for c in cells[te]])
        accs.append(float(np.mean(pred == y[te])))
    return float(np.mean(accs)) if accs else float("nan")


def trial_index_control(
    trials: pd.DataFrame,
    *,
    target_col: str = "video_id",
    feature_cols: Sequence[str] = (
        "within_subj_idx",
        "sequence_id",
        "presentation_number",
    ),
    split: str = "subject",
    n_splits: int = 5,
    n_perm: int = 200,
    seed: int = 0,
    max_rows: Optional[int] = None,
    estimator: str = "cell_mode",
) -> Dict[str, Any]:
    """Predict the stimulus label from timing metadata alone. Must be chance.

    Parameters
    ----------
    split : ``"subject"`` -- train on some subjects, test on others.  This is
        the decisive variant: if the presentation order is shared across the 80
        participants, trial index predicts the video across subjects and every
        "decoding" result in the paper is a clock reading.
        ``"sequence"`` -- hold out whole sequences within subject (catches
        within-subject order structure such as blocked orientation).
    estimator : ``"cell_mode"`` (default) is an exact modal-label lookup over
        the integer timing cell -- the most powerful decoder available for this
        particular confound, and cheap enough to permute.  ``"gbm"`` fits a
        gradient-boosting classifier instead; it is slower and *weaker* here, so
        use it only as a cross-check.
    n_perm : permutations of the timing cells across base-video blocks (see
        :func:`_block_permute_cells`).  Note this is **not** the coherent
        video-relabelling used elsewhere in the package: that shuffle is
        degenerate for this test, because a cell -> label lookup survives any
        relabelling intact.

    Returns
    -------
    dict with ``accuracy``, ``chance``, ``acc_ci_lo``/``acc_ci_hi`` (exact
    binomial), ``p_value``, ``null_mean``, ``null_sd``, ``verdict``.  The
    verdict is driven by the CI-vs-chance comparison, with the permutation
    p-value as corroboration.
    """
    from .stats import binomial_ci
    df = trials.copy()
    missing = [c for c in list(feature_cols) + [target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"trials is missing columns {missing}")
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)

    if split == "subject":
        group_col = "subject_id"
    elif split == "sequence":
        df["_seq_key"] = (
            df["subject_id"].astype(str) + "_" + df["sequence_id"].astype(str)
        )
        group_col = "_seq_key"
    else:
        raise ValueError("split must be 'subject' or 'sequence'")
    if group_col not in df.columns:
        raise KeyError(f"trials is missing the grouping column {group_col!r}")

    y = pd.Categorical(df[target_col]).codes.astype(np.int64)
    groups = df[group_col].to_numpy()
    n_classes = int(np.unique(y).size)
    # chance is the better of uniform guessing and always predicting the
    # majority class; with balanced videos the two coincide
    majority = float(np.bincount(y).max() / y.size)
    chance = max(1.0 / n_classes, majority)

    # one integer key per distinct timing cell (floats rounded to 0.1)
    keys = []
    for col in feature_cols:
        v = df[col].to_numpy()
        keys.append(np.round(v.astype(np.float64), 1) if v.dtype.kind == "f" else v)
    cells = pd.factorize(pd.MultiIndex.from_arrays(keys), sort=False)[0].astype(np.int64)

    if estimator == "cell_mode":

        def _cv(cells_use: np.ndarray) -> float:
            return _cell_mode_cv_accuracy(cells_use, y, groups, n_splits)

    elif estimator == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupKFold

        x_all = df[list(feature_cols)].to_numpy(dtype=np.float64)
        cell_to_row = {c: i for i, c in enumerate(cells)}

        def _cv(cells_use: np.ndarray) -> float:
            x = x_all[[cell_to_row[c] for c in cells_use]]
            splitter = GroupKFold(n_splits=min(n_splits, int(np.unique(groups).size)))
            accs = []
            for tr, te in splitter.split(x, y, groups):
                clf = HistGradientBoostingClassifier(
                    max_iter=80, learning_rate=0.15, random_state=seed
                ).fit(x[tr], y[tr])
                accs.append(float(np.mean(clf.predict(x[te]) == y[te])))
            return float(np.mean(accs))

    else:
        raise ValueError("estimator must be 'cell_mode' or 'gbm'")

    acc = _cv(cells)
    ci_lo, ci_hi = binomial_ci(int(round(acc * len(df))), int(len(df)))

    # Null: permute the timing cells across base-video blocks within subject.
    videos = (
        df["video_id"].to_numpy() if "video_id" in df.columns else df[target_col].to_numpy()
    )
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    exact_null = True
    for i in range(n_perm):
        permuted_cells, exact = _block_permute_cells(
            cells, videos, df["subject_id"].to_numpy(), rng
        )
        exact_null &= exact
        null[i] = _cv(permuted_cells)
    p = permutation_pvalue(acc, null, tail="greater")

    leaks = ci_lo > chance
    verdict = (
        "FAIL -- timing metadata predicts the label "
        f"({acc*100:.2f}% vs chance {chance*100:.2f}%, 95% CI lower bound "
        f"{ci_lo*100:.2f}%). Every decoding claim must regress out trial index / "
        "sequence id, and splitting by sequence is mandatory."
        if leaks
        else f"PASS -- at chance ({acc*100:.2f}% vs {chance*100:.2f}%, "
             f"95% CI [{ci_lo*100:.2f}, {ci_hi*100:.2f}])"
    )
    return {
        "target": target_col,
        "split": split,
        "estimator": estimator,
        "accuracy": acc,
        "chance": chance,
        "acc_ci_lo": ci_lo,
        "acc_ci_hi": ci_hi,
        "n_classes": n_classes,
        "p_value": p,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)) if n_perm > 1 else float("nan"),
        "n_perm": n_perm,
        "null_is_exact_block_permutation": bool(exact_null),
        "features": list(feature_cols),
        "leaks": bool(leaks),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# (c) + (d) ocular surrogate and frontal ablation
# --------------------------------------------------------------------------- #
#: The endpoint on which the ocular / ablation arms are compared.  ``trial_type``
#: is resolved per call by :func:`_resolve_trial_type` because it depends on
#: ``cfg.pseudo_k`` and on whether the split had enough repeats to build
#: pseudo-trials at all.
PRIMARY_PROBE_ENDPOINT: Dict[str, str] = {
    "direction": "eeg2vid",
    "gallery": "nway18",
    "metric": "top1",
}

#: ``constancy`` at or above this means the probe predicted (numerically) the
#: same embedding for every trial; its retrieval score is then a fixed property
#: of the gallery, not a measurement of the EEG.
DEGENERATE_CONSTANCY: float = 0.999


def _probe_retrieval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    test_item_id: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_item_ids: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    subject_ids_test: Optional[np.ndarray],
    tag: str,
    probe: Optional[RidgeAlignmentProbe] = None,
    cfg: Optional[RetrievalConfig] = None,
    arm_cache: Optional[Dict[Any, Tuple[pd.DataFrame, Dict[str, float]]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fit one arm and return its retrieval frame plus the probe diagnostics.

    The diagnostics travel with the frame, not discarded, because they are what
    tell the caller whether the arm's number means anything (see
    :meth:`RidgeAlignmentProbe._diagnose`).

    ``arm_cache`` memoises ``(tag, fingerprint of each input array)`` so that an
    arm appearing in two batteries -- ``full_eeg`` is fitted by both
    :func:`ocular_control` and :func:`frontal_ablation_sensitivity` -- costs one
    fit instead of two.  The probe is deterministic given its inputs, so this
    changes runtime only.  The fingerprint carries each array's shape, dtype and
    a strided checksum, so a cache accidentally shared across different data
    misses rather than being silently reused.
    """
    probe = probe or RidgeAlignmentProbe()
    key = None
    if arm_cache is not None:
        key = (tag, _fingerprint(x_train), _fingerprint(x_test),
               _fingerprint(np.asarray(y_train)), _fingerprint(np.asarray(gallery_emb)))
        hit = arm_cache.get(key)
        if hit is not None:
            return hit[0].copy(), dict(hit[1])
    df = probe.fit_evaluate(
        x_train, y_train, x_test, test_item_id, gallery_emb, gallery_item_ids,
        groups=groups, subject_ids_test=subject_ids_test, cfg=cfg, tag=tag,
    )
    diag = dict(probe.diagnostics_)
    if arm_cache is not None and key is not None:
        arm_cache[key] = (df.copy(), dict(diag))
    return df, diag


def _fingerprint(a: np.ndarray) -> Tuple[Any, ...]:
    """Cheap, deterministic identity for a float array (shape + strided sums).

    Uses ``.flat[::step]`` rather than ``reshape(-1)`` so a non-contiguous view
    is sampled in place instead of triggering a full copy of a multi-GB array.
    """
    arr = np.asarray(a)
    step = max(1, arr.size // 4096)
    s = np.asarray(arr.flat[::step], dtype=np.float64)
    return (arr.shape, arr.dtype.str, float(s.sum()), float((s * s).sum()), int(s.size))


def _resolve_trial_type(table: pd.DataFrame, cfg: Optional[RetrievalConfig]) -> str:
    """Pick the one ``trial_type`` every arm is compared at.

    :func:`evaluate_retrieval` emits ``single`` *and* ``pseudo{k}`` rows for the
    same (subject, direction, gallery, metric).  Selecting the endpoint without
    naming ``trial_type`` therefore returns two rows per arm, and folding that
    into a ``{probe: value}`` dict silently keeps whichever happened to come
    last -- and compares arms at *different* trial types whenever one arm had
    too few repeats for pseudo-trials.
    """
    pseudo = f"pseudo{(cfg or RetrievalConfig()).pseudo_k}"
    if "trial_type" not in table.columns or table.empty:
        return pseudo
    have = set(table["trial_type"].unique())
    if pseudo not in have:
        return "single"
    arms = set(table["probe"].unique())
    with_pseudo = set(table.loc[table["trial_type"] == pseudo, "probe"].unique())
    if arms - with_pseudo:
        warnings.warn(
            f"arms {sorted(arms - with_pseudo)} produced no {pseudo} rows; "
            "falling back to single-trial for ALL arms so the comparison stays "
            "like-for-like",
            RuntimeWarning,
        )
        return "single"
    return pseudo


def _endpoint_rows(
    table: pd.DataFrame,
    *,
    trial_type: str,
    variant: Optional[str],
    subject_id: Any = "pooled",
    endpoint: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """The endpoint rows, with the ``variant`` column resolved explicitly."""
    ep = dict(PRIMARY_PROBE_ENDPOINT if endpoint is None else endpoint)
    sel = table[
        (table["subject_id"] == subject_id)
        & (table["direction"] == ep["direction"])
        & (table["gallery"] == ep["gallery"])
        & (table["metric"] == ep["metric"])
        & (table["trial_type"] == trial_type)
    ]
    if "variant" not in table.columns:
        if variant not in (None, "with_mean"):
            raise KeyError(
                f"variant={variant!r} requested but the table has no 'variant' "
                "column (it came from a pre-variant fit_evaluate)"
            )
        return sel
    have = sorted(set(sel["variant"].astype(str)))
    if variant is None:
        if len(have) > 1:
            raise ValueError(
                f"the table holds prediction variants {have}; pass variant=... "
                "explicitly -- selecting without it would compare arms scored "
                "with and without the intercept"
            )
        return sel
    if variant not in have and len(sel):
        raise KeyError(f"variant {variant!r} not in the table; have {have}")
    return sel[sel["variant"].astype(str) == variant]


def _arm_scores(
    table: pd.DataFrame,
    *,
    trial_type: str,
    variant: Optional[str] = None,
    subject_id: Any = "pooled",
    endpoint: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """``({arm: value}, {arm: chance})`` at exactly one endpoint row per arm.

    ``variant`` names the prediction variant (:data:`PREDICTION_VARIANTS`).  It
    is only optional when the table holds a single variant; otherwise omitting
    it raises rather than silently mixing scorers.
    """
    if table.empty:
        return {}, {}
    sel = _endpoint_rows(
        table, trial_type=trial_type, variant=variant, subject_id=subject_id,
        endpoint=endpoint,
    )
    ep = dict(PRIMARY_PROBE_ENDPOINT if endpoint is None else endpoint)
    scores: Dict[str, float] = {}
    chance: Dict[str, float] = {}
    for tag, grp in sel.groupby("probe"):
        if len(grp) != 1:
            raise ValueError(
                f"endpoint {ep} / trial_type={trial_type} / variant={variant!r} "
                f"matched {len(grp)} rows for arm {tag!r}; the endpoint "
                "selection is ambiguous"
            )
        scores[str(tag)] = float(grp["value"].iloc[0])
        chance[str(tag)] = float(grp["chance"].iloc[0])
    return scores, chance


def arm_significance(
    table: pd.DataFrame,
    *,
    variant: str = PRIMARY_PREDICTION_VARIANT,
    trial_type: str = "single",
    subject_id: Any = "pooled",
    endpoint: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-arm ``{value, n, chance, ci_lo, ci_hi, z}`` at the single-trial endpoint.

    The interval is Clopper-Pearson on ``n_queries`` correct/incorrect
    retrievals and ``z = (value - chance) / sqrt(chance (1-chance) / n)``.

    **Both are anti-conservative here and neither is the inferential unit.**
    Retrieval trials are clustered within subject and within base video, so the
    effective n is far below ``n_queries``; the pre-registered unit of inference
    is the subject x fold aggregate.  These numbers are reported only to
    separate "a handful of trials above chance" from "a stable offset", which is
    the distinction the ocular verdict actually turns on.
    """
    from .stats import binomial_ci

    if table.empty:
        return {}
    sel = _endpoint_rows(
        table, trial_type=trial_type, variant=variant, subject_id=subject_id,
        endpoint=endpoint,
    )
    out: Dict[str, Dict[str, float]] = {}
    for tag, grp in sel.groupby("probe"):
        row = grp.iloc[0]
        v = float(row["value"])
        ch = float(row["chance"])
        n = int(round(float(row.get("n_queries", np.nan))))
        lo, hi = binomial_ci(int(round(v * n)), n) if n > 0 else (np.nan, np.nan)
        se = float(np.sqrt(max(ch * (1.0 - ch), 1e-12) / max(n, 1)))
        out[str(tag)] = {
            "value": v, "chance": ch, "n_queries": float(n),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "z_vs_chance": float((v - ch) / se) if n > 0 else float("nan"),
        }
    return out


def _retention(
    full: float, abl: float, chance: float, *, n_queries: Optional[float] = None
) -> Tuple[float, bool, bool, str]:
    """``(retention, is_defined, is_stable, note)`` for "fraction that survives".

    Only **defined** when the un-ablated arm is genuinely above chance.  If
    ``full <= chance`` the expression ``(abl - chance) / (full - chance)`` is a
    ratio of two below-chance deficits: it is large and positive exactly when
    both arms are equally *bad*, and printing it as "99.9% of the signal
    survives" reports the absence of any signal as a successful ablation
    control.  That is the failure this guard exists to prevent, so the value is
    withheld (NaN) rather than rendered with a caveat.

    Only **stable** when the denominator is larger than one binomial standard
    error at chance, ``sqrt(p(1-p)/n_queries)``.  A denominator inside the noise
    turns the ratio into noise/noise -- e.g. "retains 3.75 of the margin", which
    is arithmetically correct and scientifically meaningless.  The value is still
    returned (the ablated and un-ablated scores are the reportable quantities),
    but flagged so no caller prints it as a percentage.
    """
    if not all(np.isfinite(v) for v in (full, abl, chance)):
        return float("nan"), False, False, "one of full/ablated/chance is not finite"
    margin = full - chance
    if margin <= 1e-12:
        return (
            float("nan"),
            False,
            False,
            f"undefined: the un-ablated arm is at or below chance "
            f"(full={full:.5f} <= chance={chance:.5f}), so there is no "
            f"above-chance margin to retain; the ablated arm scored {abl:.5f}",
        )
    value = float((abl - chance) / margin)
    if n_queries and np.isfinite(n_queries) and n_queries > 0:
        se = float(np.sqrt(max(chance * (1.0 - chance), 1e-12) / n_queries))
        if margin < se:
            return (
                value,
                True,
                False,
                f"UNSTABLE: the un-ablated margin ({margin:.5f}) is smaller than "
                f"one binomial SE at chance ({se:.5f}, n={int(n_queries)}), so "
                "the ratio is noise over noise -- quote the two scores "
                f"(full={full:.5f}, ablated={abl:.5f}), not this fraction",
            )
    return value, True, True, ""


def _arm_n_queries(
    table: pd.DataFrame,
    *,
    trial_type: str,
    variant: Optional[str],
    subject_id: Any = "pooled",
    endpoint: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """``{arm: n_queries}`` at the same endpoint row :func:`_arm_scores` reads."""
    if table.empty or "n_queries" not in table.columns:
        return {}
    sel = _endpoint_rows(
        table, trial_type=trial_type, variant=variant, subject_id=subject_id,
        endpoint=endpoint,
    )
    return {str(tag): float(grp["n_queries"].iloc[0]) for tag, grp in sel.groupby("probe")}


def ocular_control(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    test_item_id: ArrayLike,
    gallery_emb: ArrayLike,
    gallery_item_ids: ArrayLike,
    channel_names: Sequence[str],
    *,
    groups: Optional[ArrayLike] = None,
    subject_ids_test: Optional[ArrayLike] = None,
    times_ms: Optional[ArrayLike] = None,
    presaccadic_ms: float = 150.0,
    cfg: Optional[RetrievalConfig] = None,
    probe_factory: Optional[Callable[[], RidgeAlignmentProbe]] = None,
    primary_variant: str = PRIMARY_PREDICTION_VARIANT,
    arm_cache: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Train the same alignment pipeline on ocular surrogates only.

    Arms, all on identical splits: ``full_eeg`` (all channels) and
    ``ocular_surrogate`` (HEOG/VEOG only).  When ``times_ms`` is supplied the
    same two arms are rerun on the **pre-saccadic** window (< ``presaccadic_ms``,
    the only window in which a "not ocular" latency claim is defensible) and on
    the **sustained** window (>= ``presaccadic_ms``).  Both sides of that split
    are run because the pre-saccadic claim is a *contrast*: a pre-saccadic
    margin is only interpretable next to the sustained one.

    Every arm is scored under **both** prediction variants
    (:data:`PREDICTION_VARIANTS`) from one fit.  ``primary_variant`` selects the
    one that the top-level ``scores`` / ``margins`` / ``verdict`` are read off;
    the other is always present in ``scores_by_variant``, ``margins_by_variant``
    and ``verdict_by_variant``, and both are in the ``variant`` column of
    ``table``.  Scoring only ``with_mean`` cannot tell "no signal" from "signal
    buried under a query-independent constant" -- see :data:`PREDICTION_VARIANTS`.

    ``gallery_emb`` must be the gallery the endpoint is *defined* over.  For the
    pre-registered 18-way base-video endpoint that is the fold's 18 held-out
    base videos; passing all 90 turns ``nway18`` into an analytic draw of 17
    distractors most of which the probe was trained on, under an unchanged
    column label.

    Returns
    -------
    dict with ``table`` (tidy retrieval frame, all arms x both variants),
    ``scores`` and ``chance`` at one unambiguous endpoint, ``margins`` per
    window, ``diagnostics`` (per-arm probe degeneracy), ``verdict`` and
    ``limitation`` (:data:`OCULAR_CLAIM_LIMITATION`, always present).

    The verdict is ``UNINFORMATIVE`` -- not "does not beat" -- when the
    ``full_eeg`` arm itself fails to clear chance or the probe collapsed onto a
    constant prediction.  A control that measures nothing is evidence for
    neither side, and reporting it as a failed comparison manufactures a result.
    """
    if primary_variant not in PREDICTION_VARIANTS:
        raise ValueError(
            f"unknown primary_variant {primary_variant!r}; "
            f"choose from {list(PREDICTION_VARIANTS)}"
        )
    factory = probe_factory or (lambda: RidgeAlignmentProbe())
    x_tr = np.asarray(to_numpy(x_train), dtype=np.float32)
    x_te = np.asarray(to_numpy(x_test), dtype=np.float32)
    grp = np.asarray(to_numpy(groups)).ravel() if groups is not None else None
    subs_te = (
        np.asarray(to_numpy(subject_ids_test)).ravel()
        if subject_ids_test is not None
        else None
    )
    item_te = np.asarray(to_numpy(test_item_id)).ravel()
    g_emb = np.asarray(to_numpy(gallery_emb))
    g_ids = np.asarray(to_numpy(gallery_item_ids)).ravel()
    y_tr = np.asarray(to_numpy(y_train))

    eog_tr, _ = ocular_surrogates(x_tr, channel_names)
    eog_te, _ = ocular_surrogates(x_te, channel_names)

    windows: List[Tuple[str, Optional[np.ndarray]]] = [("", None)]
    if times_ms is not None:
        t = np.asarray(to_numpy(times_ms), dtype=np.float64).ravel()
        if t.size != x_tr.shape[2]:
            raise ValueError(
                f"times_ms has {t.size} entries but the epochs have "
                f"{x_tr.shape[2]} samples"
            )
        pre = np.flatnonzero(t < presaccadic_ms)
        sus = np.flatnonzero(t >= presaccadic_ms)
        if pre.size >= 3:
            windows.append(("_presaccadic", pre))
        if sus.size >= 3:
            windows.append(("_sustained", sus))

    frames: List[pd.DataFrame] = []
    diagnostics: Dict[str, Dict[str, float]] = {}
    for suffix, keep_t in windows:
        for tag, tr, te in (
            ("full_eeg", x_tr, x_te),
            ("ocular_surrogate", eog_tr, eog_te),
        ):
            a_tr = tr if keep_t is None else tr[:, :, keep_t]
            a_te = te if keep_t is None else te[:, :, keep_t]
            df, diag = _probe_retrieval(
                a_tr, y_tr, a_te, item_te, g_emb, g_ids, groups=grp,
                subject_ids_test=subs_te, tag=f"{tag}{suffix}",
                probe=factory(), cfg=cfg, arm_cache=arm_cache,
            )
            frames.append(df)
            diagnostics[f"{tag}{suffix}"] = diag

    table = pd.concat(frames, ignore_index=True)
    trial_type = _resolve_trial_type(table, cfg)

    scores_by_variant: Dict[str, Dict[str, float]] = {}
    chance_by_variant: Dict[str, Dict[str, float]] = {}
    margins_by_variant: Dict[str, Dict[str, float]] = {}
    for v in PREDICTION_VARIANTS:
        s_v, c_v = _arm_scores(table, trial_type=trial_type, variant=v)
        scores_by_variant[v] = s_v
        chance_by_variant[v] = c_v
        margins_by_variant[v] = {
            (suffix or "_all"): float(
                s_v.get(f"full_eeg{suffix}", np.nan)
                - s_v.get(f"ocular_surrogate{suffix}", np.nan)
            )
            for suffix, _ in windows
        }

    def _verdict_for(variant: str) -> Tuple[str, bool]:
        s_v = scores_by_variant[variant]
        c_v = chance_by_variant[variant]
        const_key = "constancy" if variant == "with_mean" else "constancy_nomean"
        ok_above = bool(
            np.isfinite(s_v.get("full_eeg", np.nan))
            and s_v.get("full_eeg", np.nan) > c_v.get("full_eeg", np.nan)
        )
        const = diagnostics.get("full_eeg", {}).get(const_key, 0.0)
        ok_disp = not bool(const >= DEGENERATE_CONSTANCY)
        m = margins_by_variant[variant].get("_all", float("nan"))
        if not (ok_above and ok_disp):
            why = []
            if not ok_above:
                why.append(
                    f"the full_eeg arm is at or below chance "
                    f"({s_v.get('full_eeg', float('nan')):.4f} vs "
                    f"{c_v.get('full_eeg', float('nan')):.4f})"
                )
            if not ok_disp:
                why.append(
                    "the linear probe collapsed onto a constant prediction "
                    f"(constancy={const:.5f})"
                )
            return (
                f"UNINFORMATIVE [{variant}] -- this control cannot adjudicate the "
                "ocular question: " + "; ".join(why)
                + ". A probe that measures nothing is evidence for neither side; "
                "do not report this as 'the EEG model does not beat the "
                "surrogate'. Rerun the ablation through the trained encoder, or "
                "report the control as under-powered."
            ), False
        if np.isfinite(m) and m > 0:
            return (
                f"[{variant}] EEG model exceeds the ocular surrogate "
                f"(+{m:.5f} at the 'all' window)"
            ), True
        return (
            f"[{variant}] EEG model does NOT exceed the ocular surrogate "
            f"({m:+.5f}) -- the alignment result is not separable from "
            "frontal/ocular activity"
        ), True

    verdict_by_variant: Dict[str, str] = {}
    informative_by_variant: Dict[str, bool] = {}
    for v in PREDICTION_VARIANTS:
        verdict_by_variant[v], informative_by_variant[v] = _verdict_for(v)

    scores = scores_by_variant[primary_variant]
    chance = chance_by_variant[primary_variant]
    margins = margins_by_variant[primary_variant]
    margin = margins.get("_all", float("nan"))
    verdict = verdict_by_variant[primary_variant]
    eeg_ok = informative_by_variant[primary_variant]

    const_key = "constancy" if primary_variant == "with_mean" else "constancy_nomean"
    above_chance = {
        tag: bool(np.isfinite(v) and v > chance.get(tag, np.nan))
        for tag, v in scores.items()
    }
    degenerate = {
        tag: bool(d.get(const_key, 0.0) >= DEGENERATE_CONSTANCY)
        for tag, d in diagnostics.items()
    }

    return {
        "table": table,
        "primary_variant": primary_variant,
        "scores": scores,
        "chance": chance,
        "scores_by_variant": scores_by_variant,
        "chance_by_variant": chance_by_variant,
        "endpoint": {**PRIMARY_PROBE_ENDPOINT, "trial_type": trial_type,
                     "subject_id": "pooled", "variant": primary_variant},
        "margin": margin,
        "margins": margins,
        "margins_by_variant": margins_by_variant,
        "arms_above_chance": above_chance,
        "diagnostics": diagnostics,
        "probe_degenerate": degenerate,
        "informative": bool(eeg_ok),
        "informative_by_variant": informative_by_variant,
        "verdict": verdict,
        "verdict_by_variant": verdict_by_variant,
        "n_gallery_items": int(g_emb.shape[0]),
        "presaccadic_ms": presaccadic_ms,
        "limitation": OCULAR_CLAIM_LIMITATION,
    }


def frontal_ablation_sensitivity(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    test_item_id: ArrayLike,
    gallery_emb: ArrayLike,
    gallery_item_ids: ArrayLike,
    channel_names: Sequence[str],
    *,
    drop_prefixes: Optional[Sequence[str]] = None,
    drop_channels: Optional[Sequence[str]] = None,
    groups: Optional[ArrayLike] = None,
    subject_ids_test: Optional[ArrayLike] = None,
    times_ms: Optional[ArrayLike] = None,
    presaccadic_ms: float = 150.0,
    cfg: Optional[RetrievalConfig] = None,
    probe_factory: Optional[Callable[[], RidgeAlignmentProbe]] = None,
    primary_variant: str = PRIMARY_PREDICTION_VARIANT,
    arm_cache: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Rerun the alignment probe with the ocular channels removed (sensitivity arm).

    Defaults to the explicit :data:`OCULAR_ABLATION_CHANNELS` list (decision
    D6), **not** to prefix matching: ``drop_prefixes=("Fp","AF","F")`` also takes
    out FC*/FT*/Fz, which turns "is this ocular?" into "is this frontal cortex?"
    and makes any accuracy drop uninterpretable.  Pass ``drop_prefixes``
    explicitly if the whole-frontal ablation is what you actually want -- the
    arms are then tagged ``frontal_ablated`` instead of ``ocular_ablated``, so a
    results table never claims an 8-channel ablation it did not run.

    ``times_ms`` adds the pre-saccadic (< ``presaccadic_ms``) and sustained
    (>= ``presaccadic_ms``) window arms, which is what the gate criterion "the
    pre-saccadic signal survives frontal-channel ablation" actually needs.

    A result that survives this ablation is robust to ocular contamination *at
    the sensor level*; it still does not exclude volume-conducted saccadic spike
    potentials at posterior sensors, which is why the ocular limitation string is
    attached here as well.

    Both prediction variants are scored (see :data:`PREDICTION_VARIANTS`) and
    ``retention`` is guarded by :func:`_retention`: it is emitted only where the
    un-ablated arm actually clears chance, because below chance the ratio is two
    negatives and reads as "the signal survived" precisely when there was none.
    As with :func:`ocular_control`, ``gallery_emb`` must be the gallery the
    endpoint is defined over (the fold's held-out base videos for ``nway18``).
    """
    if primary_variant not in PREDICTION_VARIANTS:
        raise ValueError(
            f"unknown primary_variant {primary_variant!r}; "
            f"choose from {list(PREDICTION_VARIANTS)}"
        )
    factory = probe_factory or (lambda: RidgeAlignmentProbe())
    x_tr = np.asarray(to_numpy(x_train), dtype=np.float32)
    x_te = np.asarray(to_numpy(x_test), dtype=np.float32)
    if drop_prefixes is not None and drop_channels is not None:
        raise ValueError(
            "pass drop_channels OR drop_prefixes, not both -- silently letting "
            "one win is how a whole-frontal ablation gets reported as an ocular one"
        )
    if drop_prefixes is None and drop_channels is None:
        drop_channels = OCULAR_ABLATION_CHANNELS
    if drop_channels is not None:
        low = {str(c).strip().lower() for c in drop_channels}
        keep = [c for c in channel_names if str(c).strip().lower() not in low]
        x_tr_ab, kept = select_channels(x_tr, channel_names, keep=keep)
        x_te_ab, _ = select_channels(x_te, channel_names, keep=keep)
        ablated_tag = "ocular_ablated"
        ablation_kind = "explicit-channel-list (D6)"
    else:
        x_tr_ab, kept = select_channels(x_tr, channel_names, drop_prefixes=drop_prefixes)
        x_te_ab, _ = select_channels(x_te, channel_names, drop_prefixes=drop_prefixes)
        ablated_tag = "frontal_ablated"
        ablation_kind = f"prefix-match {tuple(drop_prefixes)} (WHOLE-FRONTAL, not ocular)"

    grp = np.asarray(to_numpy(groups)).ravel() if groups is not None else None
    subs_te = (
        np.asarray(to_numpy(subject_ids_test)).ravel()
        if subject_ids_test is not None
        else None
    )
    y_tr = np.asarray(to_numpy(y_train))
    item_te = np.asarray(to_numpy(test_item_id)).ravel()
    g_emb = np.asarray(to_numpy(gallery_emb))
    g_ids = np.asarray(to_numpy(gallery_item_ids)).ravel()

    windows: List[Tuple[str, Optional[np.ndarray]]] = [("", None)]
    if times_ms is not None:
        t = np.asarray(to_numpy(times_ms), dtype=np.float64).ravel()
        if t.size != x_tr.shape[2]:
            raise ValueError(
                f"times_ms has {t.size} entries but the epochs have "
                f"{x_tr.shape[2]} samples"
            )
        pre = np.flatnonzero(t < presaccadic_ms)
        sus = np.flatnonzero(t >= presaccadic_ms)
        if pre.size >= 3:
            windows.append(("_presaccadic", pre))
        if sus.size >= 3:
            windows.append(("_sustained", sus))

    frames: List[pd.DataFrame] = []
    diagnostics: Dict[str, Dict[str, float]] = {}
    for suffix, keep_t in windows:
        for tag, tr, te in (
            ("full_eeg", x_tr, x_te),
            (ablated_tag, x_tr_ab, x_te_ab),
        ):
            a_tr = tr if keep_t is None else tr[:, :, keep_t]
            a_te = te if keep_t is None else te[:, :, keep_t]
            df, diag = _probe_retrieval(
                a_tr, y_tr, a_te, item_te, g_emb, g_ids, groups=grp,
                subject_ids_test=subs_te, tag=f"{tag}{suffix}",
                probe=factory(), cfg=cfg, arm_cache=arm_cache,
            )
            frames.append(df)
            diagnostics[f"{tag}{suffix}"] = diag

    table = pd.concat(frames, ignore_index=True)
    trial_type = _resolve_trial_type(table, cfg)

    scores_by_variant: Dict[str, Dict[str, float]] = {}
    chance_by_variant: Dict[str, Dict[str, float]] = {}
    retention_by_variant: Dict[str, Dict[str, float]] = {}
    retention_valid_by_variant: Dict[str, Dict[str, bool]] = {}
    retention_stable_by_variant: Dict[str, Dict[str, bool]] = {}
    retention_notes_by_variant: Dict[str, Dict[str, str]] = {}
    for v in PREDICTION_VARIANTS:
        s_v, c_v = _arm_scores(table, trial_type=trial_type, variant=v)
        n_v = _arm_n_queries(table, trial_type=trial_type, variant=v)
        scores_by_variant[v] = s_v
        chance_by_variant[v] = c_v
        ret, ok, stable, notes = {}, {}, {}, {}
        for suffix, _ in windows:
            key = suffix or "_all"
            value, defined, is_stable, note = _retention(
                s_v.get(f"full_eeg{suffix}", np.nan),
                s_v.get(f"{ablated_tag}{suffix}", np.nan),
                c_v.get(f"full_eeg{suffix}", np.nan),
                n_queries=n_v.get(f"full_eeg{suffix}"),
            )
            ret[key], ok[key], stable[key], notes[key] = value, defined, is_stable, note
        retention_by_variant[v] = ret
        retention_valid_by_variant[v] = ok
        retention_stable_by_variant[v] = stable
        retention_notes_by_variant[v] = notes

    def _verdict_for(variant: str) -> Tuple[str, bool]:
        s_v, c_v = scores_by_variant[variant], chance_by_variant[variant]
        const_key = "constancy" if variant == "with_mean" else "constancy_nomean"
        ok_above = bool(
            np.isfinite(s_v.get("full_eeg", np.nan))
            and s_v.get("full_eeg", np.nan) > c_v.get("full_eeg", np.nan)
        )
        const = diagnostics.get("full_eeg", {}).get(const_key, 0.0)
        ok_disp = not bool(const >= DEGENERATE_CONSTANCY)
        inform = ok_above and ok_disp
        if not inform:
            return (
                f"UNINFORMATIVE [{variant}] -- the un-ablated full_eeg arm is "
                "itself at/below chance or the probe collapsed onto a constant "
                "prediction, so 'survives ablation' cannot be tested: there is "
                "no signal to ablate. Retention is withheld (it would be a ratio "
                "of two below-chance deficits)."
            ), False
        r = retention_by_variant[variant].get("_all", float("nan"))
        stable = retention_stable_by_variant[variant].get("_all", False)
        head = (
            f"[{variant}] {ablated_tag} retains {r:.2f} of the above-chance margin"
            if stable
            else f"[{variant}] {ablated_tag} retention {r:.2f} is UNSTABLE "
                 "(denominator inside the noise) -- quote the scores, not the ratio"
        )
        return (
            f"{head} ({s_v.get(ablated_tag, float('nan')):.5f} vs full "
            f"{s_v.get('full_eeg', float('nan')):.5f}, chance "
            f"{c_v.get('full_eeg', float('nan')):.5f})"
        ), True

    verdict_by_variant: Dict[str, str] = {}
    informative_by_variant: Dict[str, bool] = {}
    for v in PREDICTION_VARIANTS:
        verdict_by_variant[v], informative_by_variant[v] = _verdict_for(v)

    scores = scores_by_variant[primary_variant]
    chance = chance_by_variant[primary_variant]
    retention = retention_by_variant[primary_variant]
    verdict = verdict_by_variant[primary_variant]
    informative = informative_by_variant[primary_variant]

    const_key = "constancy" if primary_variant == "with_mean" else "constancy_nomean"
    above_chance = {
        tag: bool(np.isfinite(v) and v > chance.get(tag, np.nan))
        for tag, v in scores.items()
    }
    degenerate = {
        tag: bool(d.get(const_key, 0.0) >= DEGENERATE_CONSTANCY)
        for tag, d in diagnostics.items()
    }

    return {
        "table": table,
        "channels_kept": kept,
        "channels_dropped": [c for c in channel_names if c not in set(kept)],
        "ablation_kind": ablation_kind,
        "ablated_tag": ablated_tag,
        "primary_variant": primary_variant,
        "endpoint": {**PRIMARY_PROBE_ENDPOINT, "trial_type": trial_type,
                     "subject_id": "pooled", "variant": primary_variant},
        "scores": scores,
        "chance": chance,
        "scores_by_variant": scores_by_variant,
        "chance_by_variant": chance_by_variant,
        "retention": retention,
        "retention_valid": retention_valid_by_variant[primary_variant],
        "retention_stable": retention_stable_by_variant[primary_variant],
        "retention_notes": retention_notes_by_variant[primary_variant],
        "retention_by_variant": retention_by_variant,
        "retention_valid_by_variant": retention_valid_by_variant,
        "retention_stable_by_variant": retention_stable_by_variant,
        "retention_notes_by_variant": retention_notes_by_variant,
        "arms_above_chance": above_chance,
        "diagnostics": diagnostics,
        "probe_degenerate": degenerate,
        "informative": bool(informative),
        "informative_by_variant": informative_by_variant,
        "verdict": verdict,
        "verdict_by_variant": verdict_by_variant,
        "n_gallery_items": int(g_emb.shape[0]),
        "presaccadic_ms": presaccadic_ms,
        "limitation": OCULAR_CLAIM_LIMITATION,
    }


# --------------------------------------------------------------------------- #
# (e) low-level video features
# --------------------------------------------------------------------------- #
LOWLEVEL_FEATURE_NAMES: Tuple[str, ...] = (
    "luminance_mean",
    "luminance_std",
    "rms_contrast_mean",
    "edge_density_mean",
    "flow_mag_mean",
    "flow_mag_std",
    "flow_energy",
    "flow_dir_hcomp",
    "flow_dir_vcomp",
    "temporal_diff_energy",
    "n_frames",
)


def compute_lowlevel_features(
    video_path: str | os.PathLike[str],
    *,
    resize_to: Tuple[int, int] = (160, 120),
    max_frames: int = 40,
) -> Dict[str, float]:
    """Per-clip low-level statistics via OpenCV (optical flow, luminance, contrast).

    Features (see :data:`LOWLEVEL_FEATURE_NAMES`): mean/std luminance, mean RMS
    contrast, Canny edge density, Farneback optical-flow magnitude mean/std,
    total flow energy, mean signed horizontal and vertical flow components
    (these two flip sign under the horizontal/vertical mirror conditions, which
    is exactly the confound the orientation analyses need to control), and the
    frame-to-frame difference energy.

    Requires ``opencv-python``; raises :class:`ImportError` with an actionable
    message otherwise.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_lowlevel_features needs OpenCV: pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {video_path}")

    grays: List[np.ndarray] = []
    try:
        while len(grays) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    finally:
        cap.release()

    if len(grays) < 2:
        raise IOError(f"video {video_path} yielded {len(grays)} frames")

    stack = np.stack(grays, axis=0)
    lum_mean = float(stack.mean())
    lum_std = float(stack.mean(axis=(1, 2)).std())
    rms_contrast = float(np.mean(stack.std(axis=(1, 2))))

    edges = [
        float(np.mean(cv2.Canny((g * 255).astype(np.uint8), 60, 160) > 0)) for g in grays
    ]
    edge_density = float(np.mean(edges))

    mags: List[float] = []
    mags_sq: List[float] = []
    hcomp: List[float] = []
    vcomp: List[float] = []
    for a, b in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            (a * 255).astype(np.uint8), (b * 255).astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))
        mags_sq.append(float((mag ** 2).sum()))
        hcomp.append(float(flow[..., 0].mean()))
        vcomp.append(float(flow[..., 1].mean()))

    temporal_diff = float(np.mean([(np.abs(b - a)).mean() for a, b in zip(grays[:-1], grays[1:])]))

    return {
        "luminance_mean": lum_mean,
        "luminance_std": lum_std,
        "rms_contrast_mean": rms_contrast,
        "edge_density_mean": edge_density,
        "flow_mag_mean": float(np.mean(mags)),
        "flow_mag_std": float(np.std(mags)),
        "flow_energy": float(np.sum(mags_sq)),
        "flow_dir_hcomp": float(np.mean(hcomp)),
        "flow_dir_vcomp": float(np.mean(vcomp)),
        "temporal_diff_energy": temporal_diff,
        "n_frames": float(len(grays)),
    }


def build_lowlevel_matrix(
    condition_paths: Dict[int, str | os.PathLike[str]],
    *,
    cache_path: Optional[str | os.PathLike[str]] = None,
    n_conditions: Optional[int] = None,
    resize_to: Tuple[int, int] = (160, 120),
    max_frames: int = 40,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Build the (n_conditions, F) low-level feature matrix, with npz caching.

    ``condition_paths`` maps ``condition_id`` (or ``video_id``) to the mp4 path.
    Missing/failed clips become NaN rows and are reported; downstream code must
    mask them rather than imputing, because an imputed nuisance regressor
    understates the control.
    """
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        blob = np.load(cache, allow_pickle=False)
        return blob["features"], [str(s) for s in blob["names"]]

    keys = sorted(condition_paths)
    n = int(n_conditions if n_conditions is not None else (max(keys) + 1))
    names = list(LOWLEVEL_FEATURE_NAMES)
    out = np.full((n, len(names)), np.nan, dtype=np.float64)
    failures: List[Tuple[int, str]] = []
    for i, key in enumerate(keys):
        try:
            feats = compute_lowlevel_features(
                condition_paths[key], resize_to=resize_to, max_frames=max_frames
            )
            out[key] = [feats[nm] for nm in names]
        except Exception as exc:  # keep going; NaN rows are informative
            failures.append((key, str(exc)))
        if verbose and (i % 30 == 0):
            print(f"[lowlevel] {i}/{len(keys)} clips", flush=True)

    if failures:
        warnings.warn(f"low-level features failed for {len(failures)} clips: {failures[:5]}")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=out, names=np.array(names))
    return out, names


def residualize(y: ArrayLike, z: ArrayLike, *, add_intercept: bool = True) -> np.ndarray:
    """Remove the linear contribution of nuisance ``z`` (n, m) from ``y`` (n, d)."""
    y_arr = np.asarray(to_numpy(y), dtype=np.float64)
    z_arr = np.asarray(to_numpy(z), dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    if z_arr.ndim == 1:
        z_arr = z_arr[:, None]
    ok = np.isfinite(z_arr).all(axis=1) & np.isfinite(y_arr).all(axis=1)
    design = z_arr
    if add_intercept:
        design = np.column_stack([np.ones(z_arr.shape[0]), z_arr])
    out = y_arr.copy()
    d_ok = design[ok]
    coef, *_ = np.linalg.lstsq(d_ok, y_arr[ok], rcond=None)
    out[ok] = y_arr[ok] - d_ok @ coef
    out[~ok] = np.nan
    return out


def lowlevel_control_analysis(
    z_eeg: ArrayLike,
    trial_item_id: ArrayLike,
    gallery_emb: ArrayLike,
    gallery_item_ids: ArrayLike,
    lowlevel: ArrayLike,
    *,
    groups: Optional[ArrayLike] = None,
    subject_ids: Optional[ArrayLike] = None,
    cfg: Optional[RetrievalConfig] = None,
) -> Dict[str, Any]:
    """How much of the retrieval result is low-level video structure?

    The gallery is decomposed into the part the low-level features can explain
    and the part they cannot, by regressing the video embedding on the
    standardised low-level features.  Both parts live in the same D-dimensional
    space as the EEG embedding, so all three arms are directly comparable --
    using the raw F-dimensional feature vectors as a gallery would not be, since
    cosine similarity against a D-dimensional EEG embedding is undefined.

    Three arms on identical queries:

    * ``semantic_gallery`` -- the real video embedding gallery;
    * ``lowlevel_fitted_gallery`` -- the *fitted* values of that regression,
      i.e. the projection of the embedding onto the span of luminance,
      contrast, edge density and optical-flow energy.  Whatever accuracy this
      reaches is reachable with no semantics at all;
    * ``residual_gallery`` -- the residual, i.e. the accuracy attributable to
      embedding structure orthogonal to those features.

    Also returns the R^2 of the video embedding explained by the low-level
    features, per embedding dimension and pooled.
    """
    cfg = cfg or RetrievalConfig(n_pseudo_resamples=10, directions=("eeg2vid",))
    g_emb = np.asarray(to_numpy(gallery_emb), dtype=np.float64)
    g_ids = np.asarray(to_numpy(gallery_item_ids)).ravel()
    ll = np.asarray(to_numpy(lowlevel), dtype=np.float64)
    if ll.shape[0] != g_emb.shape[0]:
        raise ValueError(
            f"lowlevel has {ll.shape[0]} rows but the gallery has {g_emb.shape[0]}"
        )
    mu, sd = np.nanmean(ll, axis=0), np.nanstd(ll, axis=0)
    ll_z = (ll - mu) / np.where(sd > 1e-12, sd, 1.0)
    finite_rows = np.isfinite(ll_z).all(axis=1)
    if not finite_rows.all():
        warnings.warn(
            f"{int((~finite_rows).sum())} gallery items have missing low-level "
            "features; those rows are excluded from the residual arm"
        )

    ll_filled = np.nan_to_num(ll_z, nan=0.0)
    emb_unit = l2_normalize(g_emb)
    design = np.column_stack([np.ones(ll_filled.shape[0]), ll_filled])
    beta, *_ = np.linalg.lstsq(design, emb_unit, rcond=None)
    fitted = design @ beta                    # low-level-explainable part
    resid_raw = emb_unit - fitted             # orthogonal part
    fitted_gallery = l2_normalize(fitted)
    resid = l2_normalize(resid_raw)

    frames = []
    for tag, gal in (
        ("semantic_gallery", emb_unit),
        ("lowlevel_fitted_gallery", fitted_gallery),
        ("residual_gallery", resid),
    ):
        frames.append(
            evaluate_retrieval(
                z_eeg, trial_item_id, gal, g_ids, groups=groups,
                subject_ids=subject_ids, cfg=cfg, extra_fields={"probe": tag},
            )
        )
    table = pd.concat(frames, ignore_index=True)

    # variance of the embedding explained by the low-level features
    y = emb_unit[finite_rows]
    pred = fitted[finite_rows]
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2_dim = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    return {
        "table": table,
        "r2_pooled": float(1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)),
        "r2_per_dim_mean": float(np.mean(r2_dim)),
        "r2_per_dim_max": float(np.max(r2_dim)),
        "n_items_with_features": int(finite_rows.sum()),
        "feature_names": list(LOWLEVEL_FEATURE_NAMES),
    }


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_confound_battery(
    *,
    z_eeg: Optional[ArrayLike] = None,
    trials: Optional[pd.DataFrame] = None,
    subject_ids: Optional[ArrayLike] = None,
    video_ids: Optional[ArrayLike] = None,
    alignment_score: Optional[float] = None,
    alignment_chance: float = 1.0 / 18,
    epochs_train: Optional[ArrayLike] = None,
    epochs_test: Optional[ArrayLike] = None,
    y_train: Optional[ArrayLike] = None,
    test_item_id: Optional[ArrayLike] = None,
    gallery_emb: Optional[ArrayLike] = None,
    gallery_item_ids: Optional[ArrayLike] = None,
    channel_names: Optional[Sequence[str]] = None,
    groups: Optional[ArrayLike] = None,
    times_ms: Optional[ArrayLike] = None,
    lowlevel: Optional[ArrayLike] = None,
    epochs_subject_ids_test: Optional[ArrayLike] = None,
    cfg: Optional[RetrievalConfig] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run whichever probes the supplied inputs allow, and collect the caveats.

    Every probe is optional: pass the inputs you have.  The returned dict always
    carries ``"caveats"`` (a list of named limitations that must appear in the
    manuscript body) and ``"skipped"`` (probes that could not run and why), so a
    partially-run battery is never mistaken for a clean one.

    Two distinct subject-id vectors are accepted on purpose:
    ``subject_ids`` aligns with ``z_eeg`` (the learned embeddings of the test
    trials) and ``epochs_subject_ids_test`` aligns with ``epochs_test``.  They
    coincide in the usual setup, but silently reusing one for the other when
    they do not is a real source of scrambled per-subject numbers, so the second
    must be passed explicitly.
    """
    epoch_subs = (
        epochs_subject_ids_test if epochs_subject_ids_test is not None else subject_ids
    )
    out: Dict[str, Any] = {"caveats": [], "skipped": {}}

    if z_eeg is not None and subject_ids is not None and alignment_score is not None:
        out["identity"] = subject_identity_probe(
            z_eeg, subject_ids, alignment_score,
            alignment_chance=alignment_chance, video_ids=video_ids, seed=seed,
        )
    else:
        out["skipped"]["identity"] = (
            "needs z_eeg, subject_ids and alignment_score (identity is never "
            "reported alone)"
        )

    if trials is not None:
        try:
            out["trial_index_subject_split"] = trial_index_control(
                trials, split="subject", seed=seed
            )
            out["trial_index_sequence_split"] = trial_index_control(
                trials, split="sequence", seed=seed
            )
        except Exception as exc:
            out["skipped"]["trial_index"] = str(exc)
    else:
        out["skipped"]["trial_index"] = "needs the canonical trial table"

    have_probe_inputs = all(
        v is not None
        for v in (epochs_train, epochs_test, y_train, test_item_id,
                  gallery_emb, gallery_item_ids, channel_names)
    )
    if have_probe_inputs:
        out["ocular"] = ocular_control(
            epochs_train, y_train, epochs_test, test_item_id, gallery_emb,
            gallery_item_ids, channel_names, groups=groups,
            subject_ids_test=epoch_subs, times_ms=times_ms, cfg=cfg,
        )
        out["caveats"].append(OCULAR_CLAIM_LIMITATION)
        out["frontal_ablation"] = frontal_ablation_sensitivity(
            epochs_train, y_train, epochs_test, test_item_id, gallery_emb,
            gallery_item_ids, channel_names, groups=groups,
            subject_ids_test=epoch_subs, cfg=cfg,
        )
    else:
        out["skipped"]["ocular_and_ablation"] = (
            "needs epochs_train/epochs_test/y_train/test_item_id/gallery/channel_names"
        )

    if lowlevel is not None and z_eeg is not None and gallery_emb is not None:
        out["lowlevel"] = lowlevel_control_analysis(
            z_eeg, test_item_id, gallery_emb, gallery_item_ids, lowlevel,
            groups=groups, subject_ids=subject_ids, cfg=cfg,
        )
    else:
        out["skipped"]["lowlevel"] = "needs the per-condition low-level matrix"

    return out
