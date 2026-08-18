"""Dependency-light readers for the frozen TACTUS on-disk contract.

Everything here speaks *files*, not Python APIs.  The shared interface contract
freezes three artifacts:

* **A** ``data/derived/trials.parquet``      -- one row per kept non-target trial
* **B** ``data/derived/epochs/sub-XX_{window}.npy`` -- ``(n_trials, 64, T)`` float32
* **C** ``data/derived/video_emb/{tag}.npz`` -- ``cond_emb (360, D)``, ``base_emb (90, D)``

The trainer (:mod:`tactus.train.trainer`) and the four classical baselines in
:mod:`tactus.baselines` all read those three artifacts through this module, so
they stay runnable even while :mod:`tactus.data.dataset` is in flux.  Only
stdlib + numpy + pandas are imported at module level -- ``torch`` is never
imported here, which is what lets ``linear_mvpa`` run on a CPU-only box.

Path constants are re-exported from :mod:`tactus.data` (the single source of
truth); the fallbacks exist only so this module can be imported from a partially
checked-out tree.

Fold-safety note
----------------
Two per-subject transforms are fit here: Euclidean Alignment (EA) whitening and
robust per-channel scaling.  Both are **label-free**, but they are still fit on
a restricted trial set (see :class:`SubjectTransform`):

* a *training* subject is fit on its training trials only;
* a *held-out* subject is fit on its own trials (standard transductive EA, He &
  Wu 2020), optionally capped at ``max_trials`` to emulate the realistic
  "new subject, k calibration trials" enrollment protocol.

Neither uses stimulus labels, so neither leaks the retrieval target; the cap
exists because "we silently used all 576 test trials to whiten" is exactly the
kind of unstated transduction a reviewer should not have to discover.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:  # single source of truth
    from tactus.data import (  # type: ignore
        CHANNELS_JSON,
        DERIVED_DIR,
        EPOCH_DIR,
        LABEL_MAPS_JSON,
        N_CHANNELS,
        N_CONDITIONS,
        N_ORIENTATIONS,
        N_BASE_VIDEOS,
        SFREQ_TARGET,
        TRIALS_PARQUET,
        VIDEO_EMB_DIR,
        WINDOWS,
        epoch_path,
        window_spec,
    )

    REPO_ROOT = Path(__file__).resolve().parents[1]
except Exception:  # pragma: no cover - partial checkout fallback
    REPO_ROOT = Path(os.environ.get("TACTUS_ROOT", Path(__file__).resolve().parents[1]))
    _DATA_ROOT = Path(os.environ.get("TACTUS_DATA_ROOT", REPO_ROOT / "data"))
    DERIVED_DIR = _DATA_ROOT / "derived"
    EPOCH_DIR = DERIVED_DIR / "epochs"
    VIDEO_EMB_DIR = DERIVED_DIR / "video_emb"
    TRIALS_PARQUET = DERIVED_DIR / "trials.parquet"
    CHANNELS_JSON = DERIVED_DIR / "channels.json"
    LABEL_MAPS_JSON = DERIVED_DIR / "label_maps.json"
    N_CHANNELS, N_BASE_VIDEOS, N_ORIENTATIONS, N_CONDITIONS = 64, 90, 4, 360
    SFREQ_TARGET = 200.0
    WINDOWS = {"w0600": None, "wm100_800": None}

    def epoch_path(subject_id: int, window: str = "w0600", root: Path | None = None) -> Path:
        root = EPOCH_DIR if root is None else Path(root)
        return root / f"sub-{int(subject_id):02d}_{window}.npy"

    def window_spec(window: str):  # type: ignore[misc]
        raise KeyError(window)


__all__ = [
    "REPO_ROOT",
    "DERIVED_DIR",
    "WINDOW_TIMES_MS",
    "CATEGORICAL_ATTRS",
    "CONTINUOUS_ATTRS",
    "window_times_ms",
    "load_trials",
    "load_channels",
    "load_label_maps",
    "attach_label_ids",
    "continuous_stats",
    "apply_continuous_stats",
    "EpochStore",
    "VideoEmbeddings",
    "load_video_emb",
    "SubjectTransform",
    "fit_subject_transforms",
    "condition_averages",
    "split_half_reliability",
    "results_dir",
    "atomic_write_json",
    "hash_obj",
]

#: Nominal window onsets in ms; times are ``tmin + arange(T) / 200 Hz``.
WINDOW_TIMES_MS: Dict[str, tuple] = {"w0600": (0.0, 120), "wm100_800": (-100.0, 180)}

#: Categorical stimulus attributes carried in the trial table (contract A).
CATEGORICAL_ATTRS = ("material", "touch_type", "toucher", "object", "approaching")
#: Continuous stimulus attributes (z-scored before they reach a loss).
CONTINUOUS_ATTRS = ("valence", "arousal", "threat")


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def results_dir(root: Path | str | None = None) -> Path:
    """``<repo>/results`` (override with ``TACTUS_RESULTS_DIR``)."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("TACTUS_RESULTS_DIR", REPO_ROOT / "results"))


def hash_obj(obj: Any, n: int = 12) -> str:
    """Stable short hash of any JSON-able object (used for cache keys)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:n]


def atomic_write_json(path: Path | str, obj: Any, indent: int = 2) -> Path:
    """Write JSON via a temp file + ``os.replace`` so a kill never truncates it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    os.replace(tmp, path)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def window_times_ms(window: str) -> np.ndarray:
    """Sample times in milliseconds for an epoch window."""
    try:
        spec = window_spec(window)
        return spec.tmin * 1e3 + np.arange(spec.n_times) * (1e3 / spec.sfreq)
    except Exception:
        tmin, n_times = WINDOW_TIMES_MS[window]
        return tmin + np.arange(n_times) * (1e3 / SFREQ_TARGET)


# --------------------------------------------------------------------------- #
# contract A -- trial table
# --------------------------------------------------------------------------- #


def load_trials(
    path: Path | str | None = None,
    *,
    subjects: Sequence[int] | None = None,
    exclude_post_target: bool = True,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load ``trials.parquet`` (contract A).

    Parameters
    ----------
    subjects
        Keep only these ``subject_id`` values (1..80).  ``None`` keeps all.
    exclude_post_target
        Drop trials whose predecessor was a target.  Those carry the button-press
        motor potential and are excluded from **every** analysis by decree; the
        column is retained in the table purely so the exclusion can be audited.
    columns
        Optional column subset (pushed down to the parquet reader).

    Returns
    -------
    DataFrame indexed 0..n-1, sorted by ``(subject_id, within_subj_idx)`` so it
    aligns row-for-row with the per-subject epoch memmaps.
    """
    path = Path(path) if path is not None else Path(TRIALS_PARQUET)
    if not path.exists():
        raise FileNotFoundError(
            f"trial table not found at {path}. Build it first "
            f"(`make preprocess` / `python -m tactus.data.events`)."
        )
    need = None
    if columns is not None:
        need = sorted(set(columns) | {"subject_id", "within_subj_idx", "trial_uid"})
    df = pd.read_parquet(path, columns=need)
    if exclude_post_target and "is_post_target" in df.columns:
        n0 = len(df)
        df = df.loc[~df["is_post_target"].astype(bool)]
        log.info("dropped %d post-target trials (%.2f%%)", n0 - len(df), 100 * (n0 - len(df)) / max(n0, 1))
    if subjects is not None:
        df = df.loc[df["subject_id"].isin(np.asarray(subjects, dtype=np.int64))]
    df = df.sort_values(["subject_id", "within_subj_idx"]).reset_index(drop=True)
    return df


def load_channels(path: Path | str | None = None) -> list[str]:
    """The 64 BioSemi channel names, in memmap channel order (contract B)."""
    path = Path(path) if path is not None else Path(CHANNELS_JSON)
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    names = obj["channels"] if isinstance(obj, dict) else obj
    if len(names) != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} channels in {path}, got {len(names)}")
    return list(names)


# --------------------------------------------------------------------------- #
# label maps (categorical attribute -> contiguous id)
# --------------------------------------------------------------------------- #


def load_label_maps(
    trials: pd.DataFrame | None = None,
    *,
    path: Path | str | None = None,
    cache: bool = True,
) -> Dict[str, Dict[str, int]]:
    """Return ``{attr: {category: id}}`` for every categorical attribute.

    Read from ``data/derived/label_maps.json`` when present so that ids are
    identical across folds, runs and machines; otherwise derived from ``trials``
    by sorting the observed categories (deterministic) and cached.
    """
    path = Path(path) if path is not None else Path(LABEL_MAPS_JSON)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            maps = json.load(fh)
        return {k: {str(c): int(i) for c, i in v.items()} for k, v in maps.items()}
    if trials is None:
        raise FileNotFoundError(
            f"{path} does not exist and no trial table was given to derive it from."
        )
    maps: Dict[str, Dict[str, int]] = {}
    for attr in CATEGORICAL_ATTRS:
        if attr not in trials.columns:
            continue
        cats = sorted({str(c) for c in pd.Series(trials[attr]).dropna().unique()})
        maps[attr] = {c: i for i, c in enumerate(cats)}
    if cache:
        try:
            atomic_write_json(path, maps)
        except OSError:  # read-only fs -- not fatal
            log.warning("could not cache label maps to %s", path)
    return maps


def attach_label_ids(
    trials: pd.DataFrame, maps: Mapping[str, Mapping[str, int]] | None = None
) -> pd.DataFrame:
    """Add ``{attr}_id`` int columns for every categorical attribute.

    Unknown / missing categories map to ``-1`` (never silently to 0, which is a
    real class).
    """
    maps = maps if maps is not None else load_label_maps(trials)
    out = trials.copy()
    for attr, mapping in maps.items():
        if attr not in out.columns:
            continue
        out[f"{attr}_id"] = (
            out[attr].astype(str).map(lambda c: mapping.get(c, -1)).astype(np.int64)
        )
    return out


def continuous_stats(
    trials: pd.DataFrame, attrs: Sequence[str] = CONTINUOUS_ATTRS
) -> Dict[str, tuple]:
    """Mean/std of continuous attributes, computed **per unique video**.

    Weighting by trial count would make the statistics depend on how many trials
    survived artifact rejection for each video, which is a nuisance property of
    the EEG, not of the stimulus set.
    """
    stats: Dict[str, tuple] = {}
    per_video = trials.drop_duplicates(subset=["video_id"])
    for a in attrs:
        if a not in trials.columns:
            continue
        v = per_video[a].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        mu = float(v.mean()) if v.size else 0.0
        sd = float(v.std(ddof=0)) if v.size else 1.0
        stats[a] = (mu, sd if sd > 1e-8 else 1.0)
    return stats


def apply_continuous_stats(
    trials: pd.DataFrame, stats: Mapping[str, tuple], suffix: str = "_z"
) -> pd.DataFrame:
    """Add z-scored copies of the continuous attributes using fixed ``stats``."""
    out = trials.copy()
    for a, (mu, sd) in stats.items():
        if a in out.columns:
            out[a + suffix] = ((out[a].astype(np.float64) - mu) / sd).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# contract B -- epoch memmaps
# --------------------------------------------------------------------------- #


class EpochStore:
    """Lazy read-only access to the per-subject epoch memmaps.

    ``store[subject_id]`` returns the ``(n_trials, 64, T)`` memmap; ``take``
    materializes selected rows as a float32 array.  Memmaps are cached, so a
    long training run pages through the OS cache instead of re-opening files.
    """

    def __init__(self, window: str = "w0600", root: Path | str | None = None) -> None:
        self.window = str(window)
        self.root = Path(root) if root is not None else Path(EPOCH_DIR)
        self._cache: Dict[int, np.ndarray] = {}

    # -- plumbing ----------------------------------------------------------- #

    def path(self, subject_id: int) -> Path:
        return epoch_path(int(subject_id), self.window, self.root)

    def __getitem__(self, subject_id: int) -> np.ndarray:
        sid = int(subject_id)
        arr = self._cache.get(sid)
        if arr is None:
            p = self.path(sid)
            if not p.exists():
                raise FileNotFoundError(
                    f"epoch memmap missing: {p} (window={self.window}). "
                    f"Run the preprocessing stage for sub-{sid:02d}."
                )
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 3 or arr.shape[1] != N_CHANNELS:
                raise ValueError(
                    f"{p} has shape {arr.shape}; expected (n_trials, {N_CHANNELS}, T)."
                )
            self._cache[sid] = arr
        return arr

    @property
    def n_times(self) -> int:
        if self._cache:
            return int(next(iter(self._cache.values())).shape[2])
        try:
            return int(window_spec(self.window).n_times)
        except Exception:
            return int(WINDOW_TIMES_MS[self.window][1])

    def available_subjects(self) -> list[int]:
        """Subjects with an epoch file on disk for this window."""
        out = []
        for p in sorted(self.root.glob(f"sub-*_{self.window}.npy")):
            try:
                out.append(int(p.name.split("-")[1].split("_")[0]))
            except (IndexError, ValueError):
                continue
        return out

    # -- reads -------------------------------------------------------------- #

    def take(self, subject_id: int, within_idx: np.ndarray | Sequence[int]) -> np.ndarray:
        """Materialize rows ``within_idx`` of one subject as ``(n, 64, T)`` float32."""
        arr = self[subject_id]
        idx = np.asarray(within_idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, arr.shape[1], arr.shape[2]), dtype=np.float32)
        if idx.max(initial=-1) >= arr.shape[0]:
            raise IndexError(
                f"sub-{int(subject_id):02d}: within_subj_idx {int(idx.max())} out of "
                f"range for memmap with {arr.shape[0]} rows -- the trial table and "
                f"the epoch file are out of sync."
            )
        return np.asarray(arr[idx], dtype=np.float32)

    def gather(self, trials: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """Materialize every trial in ``trials`` (row order preserved).

        Beware the memory: 230k trials x 64 x 120 float32 is ~7 GB.  Use this for
        one subject at a time, or for a fold-sized subset.
        """
        n = len(trials)
        out = np.empty((n, N_CHANNELS, self.n_times), dtype=np.float32)
        pos = np.arange(n)
        for sid, grp in trials.groupby("subject_id", sort=True):
            rows = pos[trials["subject_id"].to_numpy() == sid]
            out[rows] = self.take(int(sid), grp["within_subj_idx"].to_numpy())
            if verbose:
                log.info("gathered sub-%02d (%d trials)", int(sid), len(grp))
        return out

    def close(self) -> None:
        self._cache.clear()


# --------------------------------------------------------------------------- #
# contract C -- video embeddings
# --------------------------------------------------------------------------- #


@dataclass
class VideoEmbeddings:
    """Frozen video-tower embeddings (contract C).

    ``cond_emb`` is indexed by ``condition_id`` (0..359) because the four
    orientations are *visually different stimuli*; ``base_emb`` is indexed by
    ``video_id - 1`` (original orientation only) and is what the 18-way
    base-video gallery uses.
    """

    cond_emb: np.ndarray  # (360, D) float32, L2-normalized
    base_emb: np.ndarray  # (90, D)  float32, L2-normalized
    meta: Dict[str, Any] = field(default_factory=dict)
    model_tag: str = ""

    @property
    def dim(self) -> int:
        return int(self.cond_emb.shape[1])

    def gallery(self, unit: str = "video") -> np.ndarray:
        if unit == "video":
            return self.base_emb
        if unit == "condition":
            return self.cond_emb
        raise ValueError(f"gallery unit must be 'video' or 'condition', got {unit!r}")


def load_video_emb(
    model_tag: str, root: Path | str | None = None, *, renormalize: bool = True
) -> VideoEmbeddings:
    """Load ``data/derived/video_emb/{model_tag}.npz``."""
    root = Path(root) if root is not None else Path(VIDEO_EMB_DIR)
    p = root / f"{model_tag}.npz"
    if not p.exists():
        have = sorted(q.stem for q in root.glob("*.npz"))
        raise FileNotFoundError(
            f"video embedding cache {p} not found. Available tags: {have or '(none)'}. "
            f"Build one with `make embed MODEL={model_tag}`."
        )
    with np.load(p, allow_pickle=False) as z:
        cond = np.asarray(z["cond_emb"], dtype=np.float32)
        base = np.asarray(z["base_emb"], dtype=np.float32)
        meta_raw = z["meta"] if "meta" in z.files else None
    meta: Dict[str, Any] = {}
    if meta_raw is not None:
        try:
            s = meta_raw.item() if getattr(meta_raw, "ndim", 1) == 0 else str(meta_raw)
            if isinstance(s, bytes):
                s = s.decode()
            meta = json.loads(s)
        except Exception:  # meta is advisory only
            meta = {"raw": str(meta_raw)}
    if cond.shape[0] != N_CONDITIONS or base.shape[0] != N_BASE_VIDEOS:
        raise ValueError(
            f"{p}: expected cond_emb ({N_CONDITIONS}, D) and base_emb "
            f"({N_BASE_VIDEOS}, D), got {cond.shape} and {base.shape}"
        )
    if renormalize:
        cond = cond / np.maximum(np.linalg.norm(cond, axis=1, keepdims=True), 1e-8)
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
    return VideoEmbeddings(cond_emb=cond, base_emb=base, meta=meta, model_tag=str(model_tag))


# --------------------------------------------------------------------------- #
# per-subject signal transforms (EA whitening + robust scaling)
# --------------------------------------------------------------------------- #


@dataclass
class SubjectTransform:
    """Per-subject, label-free signal normalization applied before the encoder.

    ``x -> clip( (W @ x - center) / scale, +-clamp )`` where ``W`` is the EA
    whitener (identity when EA is off), and ``center``/``scale`` are per-channel
    robust location/spread (median / IQR-based sigma).
    """

    subject_id: int
    whitener: np.ndarray | None  # (64, 64) or None
    center: np.ndarray  # (64, 1)
    scale: np.ndarray  # (64, 1)
    clamp: float = 20.0
    n_fit: int = 0

    def apply(self, x: np.ndarray) -> np.ndarray:
        """``x``: ``(64, T)`` or ``(n, 64, T)`` float32 -> same shape."""
        single = x.ndim == 2
        y = x[None] if single else x
        if self.whitener is not None:
            y = np.einsum("cd,ndt->nct", self.whitener, y, optimize=True)
        y = (y - self.center[None]) / self.scale[None]
        if self.clamp and np.isfinite(self.clamp):
            np.clip(y, -self.clamp, self.clamp, out=y)
        return y[0] if single else y

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": int(self.subject_id),
            "whitener": None if self.whitener is None else self.whitener.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "clamp": float(self.clamp),
            "n_fit": int(self.n_fit),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SubjectTransform":
        w = d.get("whitener")
        return cls(
            subject_id=int(d["subject_id"]),
            whitener=None if w is None else np.asarray(w, dtype=np.float32),
            center=np.asarray(d["center"], dtype=np.float32),
            scale=np.asarray(d["scale"], dtype=np.float32),
            clamp=float(d.get("clamp", 20.0)),
            n_fit=int(d.get("n_fit", 0)),
        )

    @classmethod
    def identity(cls, subject_id: int, n_channels: int = N_CHANNELS) -> "SubjectTransform":
        return cls(
            subject_id=int(subject_id),
            whitener=None,
            center=np.zeros((n_channels, 1), dtype=np.float32),
            scale=np.ones((n_channels, 1), dtype=np.float32),
            clamp=float("inf"),
            n_fit=0,
        )


def _ea_whitener(x: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    """Euclidean Alignment whitener ``R^{-1/2}`` (He & Wu, 2020).

    ``R`` is the arithmetic mean of the per-trial covariances (not the covariance
    of the concatenated trials -- for zero-mean epochs they coincide, but the
    mean-of-covariances form is what the EA paper specifies and it is robust to a
    handful of high-variance trials only in proportion to their count).
    """
    n, c, t = x.shape
    r = np.einsum("nct,ndt->cd", x, x, optimize=True) / float(n * t)
    r = 0.5 * (r + r.T)
    if shrinkage > 0:
        r = (1.0 - shrinkage) * r + shrinkage * (np.trace(r) / c) * np.eye(c)
    w, v = np.linalg.eigh(r.astype(np.float64))
    w = np.maximum(w, 1e-10 * float(np.max(w)) if np.max(w) > 0 else 1e-10)
    return (v @ np.diag(w ** -0.5) @ v.T).astype(np.float32)


def _robust_center_scale(x: np.ndarray) -> tuple:
    """Per-channel median and IQR-derived sigma of ``(n, 64, T)`` data."""
    flat = np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(x.shape[1], -1)
    med = np.median(flat, axis=1, keepdims=True).astype(np.float32)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    sigma = ((q75 - q25) / 1.349).astype(np.float32).reshape(-1, 1)
    bad = ~np.isfinite(sigma) | (sigma < 1e-8)
    if bad.any():  # a flat channel would otherwise explode to +-clamp
        good = sigma[~bad]
        sigma[bad] = float(np.median(good)) if good.size else 1.0
    return med, sigma


def fit_subject_transforms(
    trials: pd.DataFrame,
    store: EpochStore,
    *,
    ea: bool = True,
    ea_shrinkage: float = 0.05,
    robust_scale: bool = True,
    clamp: float = 20.0,
    max_trials: int | None = 512,
    min_trials: int = 32,
    seed: int = 0,
    cache_path: Path | str | None = None,
) -> Dict[int, SubjectTransform]:
    """Fit one :class:`SubjectTransform` per subject present in ``trials``.

    ``trials`` must already be restricted to the trials the transform is allowed
    to see (train trials for a training subject; the held-out subject's own
    trials -- or its first ``max_trials`` calibration trials -- for a new
    subject).  Results are cached to ``cache_path`` (JSON) and reused on resume.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        out = {int(k): SubjectTransform.from_dict(v) for k, v in blob.items()}
        if set(out) >= set(int(s) for s in trials["subject_id"].unique()):
            log.info("loaded %d cached subject transforms from %s", len(out), cache_path)
            return out

    rng = np.random.default_rng(seed)
    out: Dict[int, SubjectTransform] = {}
    for sid, grp in trials.groupby("subject_id", sort=True):
        sid = int(sid)
        idx = grp["within_subj_idx"].to_numpy(dtype=np.int64)
        if max_trials is not None and idx.size > max_trials:
            idx = np.sort(rng.choice(idx, size=int(max_trials), replace=False))
        if idx.size < min_trials:
            log.warning(
                "sub-%02d: only %d trials available to fit the signal transform "
                "(min %d) -- falling back to identity.",
                sid, idx.size, min_trials,
            )
            out[sid] = SubjectTransform.identity(sid)
            continue
        x = store.take(sid, idx)
        w = _ea_whitener(x, shrinkage=ea_shrinkage) if ea else None
        xw = np.einsum("cd,ndt->nct", w, x, optimize=True) if w is not None else x
        if robust_scale:
            center, scale = _robust_center_scale(xw)
        else:
            center = np.zeros((x.shape[1], 1), dtype=np.float32)
            scale = np.ones((x.shape[1], 1), dtype=np.float32)
        out[sid] = SubjectTransform(
            subject_id=sid, whitener=w, center=center, scale=scale,
            clamp=float(clamp), n_fit=int(idx.size),
        )
    if cache_path is not None:
        atomic_write_json(cache_path, {str(k): v.to_dict() for k, v in out.items()})
    return out


# --------------------------------------------------------------------------- #
# condition averages + split-half reliability (CorrCA / SRM / noise ceiling)
# --------------------------------------------------------------------------- #


def condition_averages(
    trials: pd.DataFrame,
    store: EpochStore,
    *,
    subject_id: int,
    transform: SubjectTransform | None = None,
    split: str = "all",
    n_conditions: int = N_CONDITIONS,
    decim: int = 1,
    cache_dir: Path | str | None = None,
) -> tuple:
    """Trial-averaged response per condition for one subject.

    Returns ``(avg, counts)`` where ``avg`` is ``(n_conditions, 64, T')`` float32
    with ``NaN`` for conditions that have no surviving trial, and ``counts`` is
    ``(n_conditions,)`` int32.

    ``split`` selects repeats for split-half reliability: ``"all"``, ``"odd"`` or
    ``"even"`` on ``presentation_number``.
    """
    if split not in {"all", "odd", "even"}:
        raise ValueError(f"split must be all|odd|even, got {split!r}")
    cache = None
    source_key = ""
    if cache_dir is not None:
        cache = Path(cache_dir) / f"sub-{int(subject_id):02d}_{store.window}_{split}_d{decim}.npz"
        # The key holds the parameters but says nothing about the *input*, so a
        # reprocessed subject silently kept its old averages: sub-17 was
        # re-epoched with its bad channel interpolated and CorrCA came back
        # byte-identical, because this file had not moved.  Stamp the source
        # memmap's (size, mtime) into the cache and treat a mismatch as a miss.
        try:
            st = store.path(int(subject_id)).stat()
            source_key = f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            source_key = ""
        if cache.exists():
            with np.load(cache) as z:
                cached_key = str(z["source_key"]) if "source_key" in z else ""
                if cached_key == source_key and source_key:
                    return (np.asarray(z["avg"], np.float32),
                            np.asarray(z["counts"], np.int32))
            log.info("condition_averages: sub-%02d cache is stale (source %s, "
                     "cached %s) -- recomputing", int(subject_id),
                     source_key or "?", cached_key or "unstamped")

    sub = trials.loc[trials["subject_id"] == int(subject_id)]
    if split != "all":
        parity = sub["presentation_number"].to_numpy(dtype=np.int64) % 2
        sub = sub.loc[parity == (1 if split == "odd" else 0)]
    t_full = store.n_times
    t_out = int(np.ceil(t_full / decim))
    avg = np.full((n_conditions, N_CHANNELS, t_out), np.nan, dtype=np.float32)
    counts = np.zeros(n_conditions, dtype=np.int32)
    if len(sub) == 0:
        return avg, counts

    x = store.take(int(subject_id), sub["within_subj_idx"].to_numpy())
    if transform is not None:
        x = transform.apply(x)
    if decim > 1:
        x = x[:, :, ::decim]
    cond = sub["condition_id"].to_numpy(dtype=np.int64)
    for c in np.unique(cond):
        m = cond == c
        avg[c] = x[m].mean(axis=0)
        counts[c] = int(m.sum())
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, avg=avg, counts=counts, source_key=source_key)
    return avg, counts


def split_half_reliability(
    odd: np.ndarray, even: np.ndarray, *, spearman_brown: bool = True
) -> float:
    """Correlation between two half-averages, flattened over channels x time.

    With ``spearman_brown`` the half-vs-half correlation is stepped up to the
    reliability of the *full* average, which is the quantity that belongs in the
    denominator of a noise ceiling.  ``NaN`` conditions are dropped pairwise.
    """
    a, b = np.asarray(odd, np.float64).ravel(), np.asarray(even, np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    r = float(a @ b / denom)
    if spearman_brown:
        r = 2.0 * r / (1.0 + r) if abs(1.0 + r) > 1e-12 else float("nan")
    return r
