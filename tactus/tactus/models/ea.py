"""Euclidean Alignment (EA): per-subject whitening by the inverse square root of the
mean trial covariance.

He & Wu (2020), *Transfer learning for BCIs: A Euclidean space data alignment
approach*; systematic evaluation in Junqueira et al. (2024, arXiv:2401.10746).

For subject ``s`` with trials ``X_i in R^{C x T}``::

    Rbar_s = (1/N) sum_i  X_i X_i^T / T          # mean spatial covariance
    W_s    = Rbar_s^{-1/2}                       # symmetric inverse square root
    X_i'   = W_s X_i                             # whitened trial

After the transform every subject's mean covariance is the identity, which removes
the dominant source of between-subject variability (electrode impedance, cap fit,
head geometry, reference) without touching labels.  EA is unsupervised: it uses no
trial labels, only the covariance structure.

Two things about *this* dataset make the defaults here non-standard, and both matter:

1. **The derivatives are common-average referenced, so the 64x64 covariance is
   rank 63.**  A naive ``R^{-1/2}`` divides by a numerically-zero eigenvalue and
   amplifies pure rounding noise into a full-amplitude channel.  The default
   ``rank_mode="reduce"`` therefore uses the pseudo-inverse square root: directions
   below ``rank_tol`` relative to the top eigenvalue get a zero gain instead of an
   enormous one.  Check ``EAState.rank`` after fitting -- 63 on CAR data is expected
   and correct; 64 means the data are not (exactly) CAR.
2. **Fitting must be fold-aware.**  ``fit`` is given training trials only.  What to do
   for a subject with no training trials at all (the LOSO / double-disjoint regimes) is
   a design decision that has to be pre-registered, and is exposed as ``unseen_policy``:

   ``"identity"`` (default)
       held-out subjects are not whitened.  Conservative; makes the cross-subject
       claim strictly inductive.
   ``"transductive"``
       fit ``W_s`` on the held-out subject's *own* epochs.  This is what the EA
       literature normally does -- EA needs no labels, so it is not label leakage --
       but it *is* transductive: it assumes the target subject's unlabelled data are
       available in bulk before inference.  Legitimate, and it must be stated as such.
   ``"mean_train"``
       apply the average training-subject whitener.  Available for completeness; it is
       the weakest of the three and is not recommended as a primary setting.

This module is deliberately numpy-only (``nn``-free) so alignment can be applied while
materialising epochs.  :class:`EAApplier` is a thin ``torch`` wrapper for applying
*already-fitted* whiteners inside a training loop; it never fits anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "EAState",
    "EuclideanAlignment",
    "CovarianceAccumulator",
    "EAApplier",
    "inverse_sqrt_psd",
]

ArrayLike = Union[np.ndarray, "np.memmap"]


# --------------------------------------------------------------------------------------
# linear algebra
# --------------------------------------------------------------------------------------


def inverse_sqrt_psd(
    cov: np.ndarray,
    shrinkage: float = 0.0,
    rank_mode: str = "reduce",
    rank_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Symmetric inverse square root of a PSD matrix.

    Parameters
    ----------
    cov:
        ``(C, C)`` symmetric positive semi-definite matrix.
    shrinkage:
        Ledoit-Wolf style shrink toward a scaled identity:
        ``R <- (1 - a) R + a * (tr(R) / C) I``.  ``0`` disables it.
    rank_mode:
        ``"reduce"`` -- zero the gain of eigen-directions below ``rank_tol * max_eig``
        (pseudo-inverse square root; correct for rank-deficient / CAR data).
        ``"floor"`` -- clamp those eigenvalues up to ``rank_tol * max_eig`` instead,
        keeping the transform full-rank but bounded.
    rank_tol:
        Relative eigenvalue tolerance.

    Returns
    -------
    (whitener, eigvals, rank)
        ``whitener`` is ``(C, C)`` symmetric, ``eigvals`` are the (shrunk) eigenvalues
        in ascending order, ``rank`` is the number of retained directions.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got {cov.shape}")
    if rank_mode not in ("reduce", "floor"):
        raise ValueError(f"rank_mode must be 'reduce' or 'floor', got {rank_mode!r}")
    c = cov.shape[0]
    cov = 0.5 * (cov + cov.T)  # enforce symmetry against accumulation drift
    if shrinkage > 0.0:
        mu = float(np.trace(cov)) / c
        cov = (1.0 - float(shrinkage)) * cov + float(shrinkage) * mu * np.eye(c)

    eigvals, eigvecs = np.linalg.eigh(cov)
    max_eig = float(eigvals.max()) if eigvals.size else 0.0
    if max_eig <= 0.0:
        raise ValueError("covariance has no positive eigenvalue; the input is all-zero or corrupt")
    thresh = float(rank_tol) * max_eig
    keep = eigvals > thresh
    rank = int(keep.sum())

    inv_sqrt = np.zeros_like(eigvals)
    if rank_mode == "reduce":
        inv_sqrt[keep] = 1.0 / np.sqrt(eigvals[keep])
    else:
        clamped = np.maximum(eigvals, thresh)
        inv_sqrt = 1.0 / np.sqrt(clamped)
    whitener = (eigvecs * inv_sqrt) @ eigvecs.T
    whitener = 0.5 * (whitener + whitener.T)
    return whitener, eigvals, rank


# --------------------------------------------------------------------------------------
# per-subject state
# --------------------------------------------------------------------------------------


@dataclass
class EAState:
    """Fitted alignment for one subject."""

    subject_id: int
    whitener: np.ndarray  # (C, C) float64
    mean_cov: np.ndarray  # (C, C) float64
    n_trials: int
    n_samples: int
    rank: int
    eigvals: np.ndarray
    shrinkage: float
    rank_mode: str
    rank_tol: float
    source: str = "train"  # "train" | "transductive" | "mean_train" | "identity"

    @property
    def n_channels(self) -> int:
        """Channel count."""
        return int(self.whitener.shape[0])

    @property
    def condition_number(self) -> float:
        """Condition number over the retained eigen-directions (a fit-quality flag)."""
        ev = np.sort(self.eigvals)[::-1]
        kept = ev[: self.rank]
        return float(kept[0] / kept[-1]) if kept.size and kept[-1] > 0 else float("inf")

    def apply(self, x: ArrayLike, dtype: np.dtype = np.float32) -> np.ndarray:
        """Whiten ``(C, T)`` or ``(N, C, T)`` epochs."""
        arr = np.asarray(x)
        w = self.whitener
        if arr.ndim == 2:
            return (w @ arr).astype(dtype, copy=False)
        if arr.ndim == 3:
            return np.einsum("ij,njt->nit", w, arr, optimize=True).astype(dtype, copy=False)
        raise ValueError(f"expected (C, T) or (N, C, T), got {arr.shape}")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe summary (without the matrices) for the run log."""
        return {
            "subject_id": int(self.subject_id),
            "n_trials": int(self.n_trials),
            "n_samples": int(self.n_samples),
            "n_channels": self.n_channels,
            "rank": int(self.rank),
            "condition_number": self.condition_number,
            "shrinkage": float(self.shrinkage),
            "rank_mode": self.rank_mode,
            "rank_tol": float(self.rank_tol),
            "source": self.source,
        }


class CovarianceAccumulator:
    """Streaming accumulator for the mean trial covariance of one subject.

    Lets EA be fitted over a memory-mapped epoch array without loading it all::

        acc = CovarianceAccumulator(n_channels=64)
        for chunk in chunks:
            acc.update(chunk)              # (n, 64, T)
        mean_cov = acc.mean_cov()
    """

    def __init__(self, n_channels: int = 64, per_trial_trace_norm: bool = False) -> None:
        self.n_channels = int(n_channels)
        self.per_trial_trace_norm = bool(per_trial_trace_norm)
        self._sum = np.zeros((self.n_channels, self.n_channels), dtype=np.float64)
        self._n = 0
        self._n_samples = 0

    def update(self, epochs: ArrayLike) -> "CovarianceAccumulator":
        """Add ``(N, C, T)`` (or a single ``(C, T)``) epoch block."""
        arr = np.asarray(epochs, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError(f"expected (N, C, T), got {arr.shape}")
        if arr.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {arr.shape[1]}")
        if not np.isfinite(arr).all():
            raise ValueError("non-finite values in epochs; clean or drop them before fitting EA")
        n, _, t = arr.shape
        covs = np.einsum("nct,ndt->ncd", arr, arr, optimize=True) / float(t)
        if self.per_trial_trace_norm:
            traces = np.einsum("ncc->n", covs)
            traces = np.where(traces > 0, traces, 1.0)
            covs = covs / traces[:, None, None] * self.n_channels
        self._sum += covs.sum(axis=0)
        self._n += int(n)
        self._n_samples += int(n * t)
        return self

    @property
    def n_trials(self) -> int:
        """Trials accumulated so far."""
        return self._n

    @property
    def n_samples(self) -> int:
        """Time samples accumulated so far."""
        return self._n_samples

    def mean_cov(self) -> np.ndarray:
        """The mean trial covariance ``(C, C)``."""
        if self._n == 0:
            raise ValueError("no trials accumulated")
        return self._sum / float(self._n)


# --------------------------------------------------------------------------------------
# the transform
# --------------------------------------------------------------------------------------


class EuclideanAlignment:
    """Fold-aware per-subject Euclidean Alignment.

    Usage (one instance per fold)::

        ea = EuclideanAlignment(unseen_policy="identity")
        for sid, x_train in train_epochs_by_subject.items():
            ea.fit_subject(x_train, sid)              # TRAIN trials only
        x_aligned = ea.transform(x_any, subject_ids)  # any trials, train or test

    Parameters
    ----------
    shrinkage:
        Shrink the mean covariance toward a scaled identity before inversion.
    rank_mode, rank_tol:
        See :func:`inverse_sqrt_psd`.  ``"reduce"`` is the right default for the
        common-average-referenced derivatives.
    per_trial_trace_norm:
        Normalise each trial's covariance to trace ``C`` before averaging.  Makes the
        fit robust to a handful of huge-amplitude artefact trials at the cost of
        departing from the canonical He & Wu estimator.  Worth turning on given that
        the derivatives carry no artefact rejection and no ICA.
    min_trials:
        Refuse to fit a subject with fewer than this many trials (a whitener fitted on
        a handful of trials is noise).  64 channels need well over 64 trials.
    unseen_policy:
        ``"identity" | "transductive" | "mean_train"``; see the module docstring.
        Pre-register the choice.
    dtype:
        Output dtype of :meth:`transform` (``float32`` to match contract B).
    """

    def __init__(
        self,
        shrinkage: float = 0.0,
        rank_mode: str = "reduce",
        rank_tol: float = 1e-6,
        per_trial_trace_norm: bool = False,
        min_trials: int = 128,
        unseen_policy: str = "identity",
        dtype: np.dtype = np.float32,
    ) -> None:
        if unseen_policy not in ("identity", "transductive", "mean_train"):
            raise ValueError(f"unknown unseen_policy {unseen_policy!r}")
        self.shrinkage = float(shrinkage)
        self.rank_mode = str(rank_mode)
        self.rank_tol = float(rank_tol)
        self.per_trial_trace_norm = bool(per_trial_trace_norm)
        self.min_trials = int(min_trials)
        self.unseen_policy = str(unseen_policy)
        self.dtype = dtype
        self.states: Dict[int, EAState] = {}
        self._mean_train_whitener: Optional[np.ndarray] = None

    # -- fitting ----------------------------------------------------------------------

    def fit_subject(
        self,
        epochs: ArrayLike,
        subject_id: int,
        source: str = "train",
        chunk: int = 512,
    ) -> EAState:
        """Fit one subject from ``(N, C, T)`` **training** epochs (memmap-friendly).

        ``epochs`` may be a ``np.memmap``; it is read in blocks of ``chunk`` trials and
        never materialised in full.
        """
        arr = epochs
        if arr.ndim == 2:  # a single (C, T) trial
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError(f"expected (N, C, T), got {arr.shape}")
        acc = CovarianceAccumulator(
            n_channels=int(arr.shape[1]), per_trial_trace_norm=self.per_trial_trace_norm
        )
        for start in range(0, int(arr.shape[0]), int(chunk)):
            acc.update(np.asarray(arr[start : start + int(chunk)]))
        if acc.n_trials < self.min_trials:
            raise ValueError(
                f"subject {subject_id}: only {acc.n_trials} trials (min_trials={self.min_trials}); "
                "an EA whitener fitted on this many trials is dominated by estimation noise. "
                "Lower min_trials explicitly if this is intended."
            )
        mean_cov = acc.mean_cov()
        whitener, eigvals, rank = inverse_sqrt_psd(
            mean_cov, shrinkage=self.shrinkage, rank_mode=self.rank_mode, rank_tol=self.rank_tol
        )
        state = EAState(
            subject_id=int(subject_id),
            whitener=whitener,
            mean_cov=mean_cov,
            n_trials=acc.n_trials,
            n_samples=acc.n_samples,
            rank=rank,
            eigvals=eigvals,
            shrinkage=self.shrinkage,
            rank_mode=self.rank_mode,
            rank_tol=self.rank_tol,
            source=source,
        )
        self.states[int(subject_id)] = state
        self._mean_train_whitener = None  # invalidate cache
        return state

    def fit(
        self,
        epochs: ArrayLike,
        subject_ids: Sequence[int],
        source: str = "train",
    ) -> "EuclideanAlignment":
        """Fit every subject present in ``subject_ids`` from a concatenated ``(N, C, T)`` array."""
        sids = np.asarray(subject_ids).reshape(-1)
        if sids.shape[0] != np.asarray(epochs).shape[0]:
            raise ValueError(
                f"epochs has {np.asarray(epochs).shape[0]} trials but subject_ids has {sids.shape[0]}"
            )
        for sid in np.unique(sids):
            idx = np.flatnonzero(sids == sid)
            self.fit_subject(np.asarray(epochs)[idx], int(sid), source=source)
        return self

    def fit_from_memmaps(
        self,
        trials: "Any",
        epochs_root: Union[str, Path],
        window: str = "w0600",
        train_uid: Optional[Sequence[int]] = None,
        subjects: Optional[Sequence[int]] = None,
        chunk: int = 512,
        max_trials_per_subject: Optional[int] = None,
        seed: int = 0,
    ) -> "EuclideanAlignment":
        """Fit from the on-disk epoch memmaps of contract B, using training rows only.

        Parameters
        ----------
        trials:
            The canonical trial table (contract A) as a pandas DataFrame.
        epochs_root:
            Directory holding ``sub-{ID:02d}_{window}.npy``.
        window:
            ``"w0600"`` (primary) or ``"wm100_800"`` (sensitivity).
        train_uid:
            ``trial_uid`` values of this fold's training trials.  ``None`` fits on every
            row of ``trials``, which is only correct outside a cross-validation loop.
        subjects:
            Restrict to these subjects (e.g. ``fold.train_subjects``).
        max_trials_per_subject:
            Sub-sample for speed; a few hundred trials already estimate a 64x64
            covariance well.  Sampling is seeded and reproducible.
        """
        root = Path(epochs_root)
        df = trials
        if train_uid is not None:
            df = df[df["trial_uid"].isin(np.asarray(train_uid))]
        if subjects is not None:
            df = df[df["subject_id"].isin(np.asarray(subjects))]
        if len(df) == 0:
            raise ValueError("no training trials selected; check train_uid / subjects")
        rng = np.random.default_rng(seed)
        for sid, grp in df.groupby("subject_id", sort=True):
            path = root / f"sub-{int(sid):02d}_{window}.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing epoch memmap {path}")
            mm = np.load(path, mmap_mode="r")
            idx = np.sort(grp["within_subj_idx"].to_numpy().astype(np.int64))
            if max_trials_per_subject is not None and idx.size > int(max_trials_per_subject):
                idx = np.sort(rng.choice(idx, size=int(max_trials_per_subject), replace=False))
            acc = CovarianceAccumulator(
                n_channels=int(mm.shape[1]), per_trial_trace_norm=self.per_trial_trace_norm
            )
            for start in range(0, idx.size, int(chunk)):
                acc.update(np.asarray(mm[idx[start : start + int(chunk)]]))
            if acc.n_trials < self.min_trials:
                raise ValueError(
                    f"subject {sid}: only {acc.n_trials} training trials (min_trials={self.min_trials})"
                )
            mean_cov = acc.mean_cov()
            whitener, eigvals, rank = inverse_sqrt_psd(
                mean_cov, shrinkage=self.shrinkage, rank_mode=self.rank_mode, rank_tol=self.rank_tol
            )
            self.states[int(sid)] = EAState(
                subject_id=int(sid),
                whitener=whitener,
                mean_cov=mean_cov,
                n_trials=acc.n_trials,
                n_samples=acc.n_samples,
                rank=rank,
                eigvals=eigvals,
                shrinkage=self.shrinkage,
                rank_mode=self.rank_mode,
                rank_tol=self.rank_tol,
                source="train",
            )
        self._mean_train_whitener = None
        return self

    def fit_unseen_transductive(self, epochs: ArrayLike, subject_id: int, chunk: int = 512) -> EAState:
        """Fit a held-out subject from that subject's own (unlabelled) epochs.

        Only legal when ``unseen_policy == "transductive"``; the guard exists so a
        transductive fit cannot be introduced by accident.
        """
        if self.unseen_policy != "transductive":
            raise RuntimeError(
                f"unseen_policy is {self.unseen_policy!r}; fitting a held-out subject would "
                "silently change the pre-registered protocol. Construct EuclideanAlignment("
                "unseen_policy='transductive') if that is what you intend."
            )
        return self.fit_subject(epochs, subject_id, source="transductive", chunk=chunk)

    # -- applying ---------------------------------------------------------------------

    def mean_train_whitener(self) -> np.ndarray:
        """Element-wise mean of the fitted training whiteners (for ``unseen_policy="mean_train"``)."""
        if self._mean_train_whitener is None:
            mats = [s.whitener for s in self.states.values() if s.source == "train"]
            if not mats:
                raise ValueError("no training subjects fitted")
            self._mean_train_whitener = np.mean(np.stack(mats, axis=0), axis=0)
        return self._mean_train_whitener

    def whitener_for(self, subject_id: int, n_channels: int) -> np.ndarray:
        """The ``(C, C)`` matrix applied to ``subject_id``, following ``unseen_policy``."""
        sid = int(subject_id)
        if sid in self.states:
            return self.states[sid].whitener
        if self.unseen_policy == "identity":
            return np.eye(n_channels, dtype=np.float64)
        if self.unseen_policy == "mean_train":
            return self.mean_train_whitener()
        raise KeyError(
            f"subject {sid} has no fitted EA state and unseen_policy is 'transductive'; "
            "call fit_unseen_transductive() on that subject's epochs first."
        )

    def transform(self, epochs: ArrayLike, subject_ids: Union[int, Sequence[int]]) -> np.ndarray:
        """Whiten ``(N, C, T)`` epochs given each trial's subject id (or one shared id)."""
        arr = np.asarray(epochs)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError(f"expected (N, C, T), got {arr.shape}")
        n, c, _ = arr.shape
        sids = np.full(n, int(subject_ids)) if np.isscalar(subject_ids) else np.asarray(subject_ids).reshape(-1)
        if sids.shape[0] != n:
            raise ValueError(f"epochs has {n} trials but subject_ids has {sids.shape[0]}")
        out = np.empty_like(arr, dtype=self.dtype)
        for sid in np.unique(sids):
            idx = np.flatnonzero(sids == sid)
            w = self.whitener_for(int(sid), c)
            out[idx] = np.einsum("ij,njt->nit", w, arr[idx].astype(np.float64), optimize=True).astype(
                self.dtype, copy=False
            )
        return out

    def fit_transform(
        self, epochs: ArrayLike, subject_ids: Sequence[int]
    ) -> np.ndarray:
        """Fit on ``epochs`` and immediately transform them.

        Only valid when ``epochs`` *are* the training trials; using this on a mixed
        train/test array leaks test covariance structure into the fit.
        """
        self.fit(epochs, subject_ids)
        return self.transform(epochs, subject_ids)

    # -- persistence ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Save all fitted whiteners to a single ``.npz`` (one file per fold)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sids = sorted(self.states)
        arrays: Dict[str, np.ndarray] = {
            "subject_ids": np.asarray(sids, dtype=np.int64),
            "whiteners": np.stack([self.states[s].whitener for s in sids]) if sids else np.zeros((0, 0, 0)),
            "mean_covs": np.stack([self.states[s].mean_cov for s in sids]) if sids else np.zeros((0, 0, 0)),
            "eigvals": np.stack([self.states[s].eigvals for s in sids]) if sids else np.zeros((0, 0)),
            "n_trials": np.asarray([self.states[s].n_trials for s in sids], dtype=np.int64),
            "ranks": np.asarray([self.states[s].rank for s in sids], dtype=np.int64),
        }
        meta = {
            "shrinkage": self.shrinkage,
            "rank_mode": self.rank_mode,
            "rank_tol": self.rank_tol,
            "per_trial_trace_norm": self.per_trial_trace_norm,
            "min_trials": self.min_trials,
            "unseen_policy": self.unseen_policy,
            "subjects": {str(s): self.states[s].to_dict() for s in sids},
        }
        arrays["meta"] = np.asarray(json.dumps(meta, indent=2))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EuclideanAlignment":
        """Load whiteners saved by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            obj = cls(
                shrinkage=meta["shrinkage"],
                rank_mode=meta["rank_mode"],
                rank_tol=meta["rank_tol"],
                per_trial_trace_norm=meta["per_trial_trace_norm"],
                min_trials=meta["min_trials"],
                unseen_policy=meta["unseen_policy"],
            )
            sids = z["subject_ids"]
            for i, sid in enumerate(sids.tolist()):
                info = meta["subjects"][str(int(sid))]
                obj.states[int(sid)] = EAState(
                    subject_id=int(sid),
                    whitener=z["whiteners"][i],
                    mean_cov=z["mean_covs"][i],
                    n_trials=int(z["n_trials"][i]),
                    n_samples=int(info["n_samples"]),
                    rank=int(z["ranks"][i]),
                    eigvals=z["eigvals"][i],
                    shrinkage=obj.shrinkage,
                    rank_mode=obj.rank_mode,
                    rank_tol=obj.rank_tol,
                    source=info.get("source", "train"),
                )
        return obj

    # -- reporting --------------------------------------------------------------------

    def report(self) -> List[Dict[str, Any]]:
        """Per-subject fit diagnostics; check ``rank`` and ``condition_number``."""
        return [self.states[s].to_dict() for s in sorted(self.states)]

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"EuclideanAlignment(n_fitted={len(self.states)}, shrinkage={self.shrinkage}, "
            f"rank_mode={self.rank_mode!r}, unseen_policy={self.unseen_policy!r})"
        )


# --------------------------------------------------------------------------------------
# torch applier (apply only -- never fits)
# --------------------------------------------------------------------------------------


class EAApplier:
    """Apply already-fitted EA whiteners inside a torch training loop.

    Constructed lazily so this module keeps working without torch installed.

    ``applier(x, subject_id)`` takes ``(B, C, T)`` and ``(B,)`` and returns ``(B, C, T)``;
    subjects with no fitted whitener get the identity (or the mean-train whitener, per
    ``unseen_policy``), exactly matching :meth:`EuclideanAlignment.whitener_for`.
    """

    def __init__(self, alignment: EuclideanAlignment, n_channels: int = 64, device: Any = None) -> None:
        import torch  # local import: this class is the only torch dependency in the module

        self._torch = torch
        self.alignment = alignment
        self.n_channels = int(n_channels)
        sids = sorted(alignment.states)
        max_id = max(sids) if sids else 0
        lut = torch.full((max_id + 2,), -1, dtype=torch.long)
        mats = []
        for row, sid in enumerate(sids):
            lut[sid] = row
            mats.append(torch.as_tensor(alignment.states[sid].whitener, dtype=torch.float32))
        if alignment.unseen_policy == "mean_train" and sids:
            fallback = torch.as_tensor(alignment.mean_train_whitener(), dtype=torch.float32)
        else:
            fallback = torch.eye(self.n_channels, dtype=torch.float32)
        mats.append(fallback)  # last row = fallback for unseen subjects
        self.fallback_row = len(mats) - 1
        self.lut = lut.to(device) if device is not None else lut
        self.whiteners = (
            torch.stack(mats).to(device) if device is not None else torch.stack(mats)
        )

    def to(self, device: Any) -> "EAApplier":
        """Move the whitener table to ``device``."""
        self.lut = self.lut.to(device)
        self.whiteners = self.whiteners.to(device)
        return self

    def __call__(self, x: Any, subject_id: Any) -> Any:
        torch = self._torch
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
        sid = torch.as_tensor(subject_id).reshape(-1).to(self.lut.device, torch.long)
        rows = torch.full_like(sid, self.fallback_row)
        in_range = (sid >= 0) & (sid < self.lut.numel())
        if bool(in_range.any()):
            mapped = self.lut[sid[in_range]]
            rows[in_range] = torch.where(
                mapped >= 0, mapped, torch.full_like(mapped, self.fallback_row)
            )
        w = self.whiteners.to(device=x.device, dtype=x.dtype)[rows.to(x.device)]
        return torch.bmm(w, x)
