"""Zero-shot retrieval evaluation for TACTUS (shared interface contract G).

Design notes that matter for correctness
----------------------------------------
1.  **N-way retrieval is computed in closed form, not by Monte-Carlo.**
    Once the rank ``r`` of the true item among all ``M`` gallery items is known,
    the expected top-k accuracy under uniform sampling of ``N-1`` distractors
    (without replacement, true item always present) is exactly

        P(top-k) = sum_{j=0}^{k-1} C(r-1, j) * C(M-r, N-1-j) / C(M-1, N-1)

    and the expected rank in the sub-gallery is exactly

        E[rank_N] = 1 + (N-1) * (r-1) / (M-1)

    This removes an entire class of "how many distractor draws is enough"
    reviewer questions and makes the numbers deterministic across runs.
    A Monte-Carlo path is retained (``method="mc"``) purely as a unit-test
    cross-check of the analytic path.

2.  **Within-material distractors quantify the attribute-shortcut ceiling.**
    ``groups`` is normally the 8-way material label.  If full-gallery retrieval
    is far above chance but *within-material* retrieval sits at chance, the
    "zero-shot video retrieval" result is a repackaged 8-way material code.
    We therefore report the within-group and cross-group variants next to the
    headline number, always, not on request.

3.  **Pseudo-trials average k=4 repeats of the same item within one subject.**
    Repeats are never pooled across subjects (that would be a different, much
    easier problem) and never pooled across items.

4.  Directions: ``eeg2vid`` (query = EEG trial, gallery = video embeddings) and
    ``vid2eeg`` (query = video embedding, gallery = one EEG representation per
    candidate item).  The second direction is the one that is sensitive to a
    collapsed / hubness-dominated EEG embedding space, so both are reported.

.. _centring-convention:

5.  **Centring convention (canonical definition -- DECISIONS D12).**
    *The training-set mean is subtracted from the query and from the gallery,
    and from nothing else.*  Every arm that quotes a retrieval number obeys
    this: :mod:`tactus.baselines.srm`, :mod:`tactus.baselines.linear_align`
    (``{unit}_centered``, its :data:`~tactus.baselines.linear_align.PRIMARY_VARIANT`)
    and the deep-model evaluation in :mod:`tactus.train.trainer`.  Cite this
    paragraph rather than restating it.

    Three properties are load-bearing:

    * The mean is a **training** statistic, so nothing about the held-out side
      leaks into its own scoring -- this is why SRM's convention won over
      linear_align's original one rather than the reverse.
    * Query and gallery are centred by the training mean **of their own space**.
      A ridge that predicts into condition-embedding space and a gallery of
      base-video embeddings do not share an origin, so they do not share a
      centre (``y_center`` vs ``base_center``).
    * Centring only the query is not a milder version of this -- it is a
      different and worse estimator.  Cosine against an uncentred, anisotropic
      gallery (mean pairwise cosine 0.88 on the SigLIP2 embeddings here) is
      dominated by a query-independent bias term; removing the mean from one
      side alone leaves that bias in place while stripping the signal that
      partially cancelled it, which is what pinned linear_align's headline to
      chance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import gammaln

__all__ = [
    "DEFAULT_GALLERY_SIZES",
    "RetrievalConfig",
    "to_numpy",
    "l2_normalize",
    "rank_of_true",
    "nway_topk_accuracy",
    "nway_mean_rank",
    "retrieval_metrics",
    "metrics_to_long",
    "pseudo_trial_index",
    "build_pseudo_trials",
    "evaluate_retrieval",
]

ArrayLike = Any

#: Gallery sizes required by contract G.  360 is the full-condition gallery and
#: is silently skipped whenever fewer than 360 items are available in the split.
DEFAULT_GALLERY_SIZES: Tuple[int, ...] = (2, 10, 18, 90, 360)


# --------------------------------------------------------------------------- #
# small shared numeric helpers (imported by the other tactus.eval modules)
# --------------------------------------------------------------------------- #
def to_numpy(x: ArrayLike) -> Optional[np.ndarray]:
    """Convert torch tensors / lists / pandas objects to a numpy array.

    Kept dependency-free on purpose: ``tactus.eval`` must import without torch.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "to_numpy"):  # pandas
        return np.asarray(x.to_numpy())
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation with a numerically safe denominator."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _log_comb(n: ArrayLike, k: ArrayLike) -> np.ndarray:
    """log C(n, k), broadcasting, ``-inf`` where the binomial is zero."""
    n_arr = np.asarray(n, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    shape = np.broadcast_shapes(n_arr.shape, k_arr.shape)
    nb = np.broadcast_to(n_arr, shape)
    kb = np.broadcast_to(k_arr, shape)
    out = np.full(shape, -np.inf, dtype=np.float64)
    ok = (kb >= 0) & (kb <= nb) & (nb >= 0)
    if np.any(ok):
        out[ok] = gammaln(nb[ok] + 1.0) - gammaln(kb[ok] + 1.0) - gammaln(nb[ok] - kb[ok] + 1.0)
    return out


# --------------------------------------------------------------------------- #
# ranks
# --------------------------------------------------------------------------- #
def rank_of_true(
    scores: np.ndarray,
    true_col: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """1-based rank of the true gallery item, ties counted as half.

    Parameters
    ----------
    scores : (B, M) similarity matrix, higher = more similar.
    true_col : (B,) index into the gallery for the correct item of each query.
    mask : optional (B, M) boolean.  ``False`` entries are removed from the
        gallery for that query (used for within-/cross-group galleries).  The
        true column must be ``True`` wherever it is scored.

    Returns
    -------
    (B,) float array of ranks in ``[1, n_valid]``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    b = scores.shape[0]
    rows = np.arange(b)
    true_score = scores[rows, true_col]
    if mask is None:
        greater = (scores > true_score[:, None]).sum(axis=1)
        equal = (scores == true_score[:, None]).sum(axis=1) - 1
    else:
        mask = np.asarray(mask, dtype=bool)
        greater = ((scores > true_score[:, None]) & mask).sum(axis=1)
        equal = ((scores == true_score[:, None]) & mask).sum(axis=1) - 1
    return 1.0 + greater + 0.5 * np.maximum(equal, 0)


def nway_topk_accuracy(
    ranks: ArrayLike,
    n_items: int,
    n_way: int,
    k: int = 1,
    *,
    method: str = "exact",
    n_draws: int = 500,
    seed: int = 0,
) -> float:
    """Expected top-k accuracy in an ``n_way`` sub-gallery, from full-gallery ranks.

    ``method="exact"`` uses the hypergeometric closed form (default, deterministic).
    ``method="mc"`` resamples distractors and is provided only to validate the
    closed form in tests.
    """
    r = np.asarray(ranks, dtype=np.float64)
    m = int(n_items)
    n = int(n_way)
    if r.size == 0:
        return float("nan")
    if n >= m:
        return float(np.mean(r <= k))
    if n < 2:
        raise ValueError("n_way must be >= 2")

    if method == "mc":
        rng = np.random.default_rng(seed)
        r_int = np.clip(np.rint(r).astype(np.int64), 1, m)
        better = r_int - 1  # number of gallery items ranked above the true item
        hits = np.zeros(r_int.shape[0], dtype=np.float64)
        for _ in range(n_draws):
            # hypergeometric: how many of the n-1 drawn distractors outrank truth
            drawn_better = rng.hypergeometric(better, m - 1 - better, n - 1)
            hits += (drawn_better <= (k - 1)).astype(np.float64)
        return float(np.mean(hits / n_draws))

    if method != "exact":
        raise ValueError(f"unknown method {method!r}")

    r_int = np.clip(np.rint(r), 1.0, float(m))
    log_denom = _log_comb(m - 1, n - 1)
    total = np.zeros(r_int.shape, dtype=np.float64)
    for j in range(0, int(k)):
        lp = _log_comb(r_int - 1.0, j) + _log_comb(m - r_int, n - 1 - j) - log_denom
        total += np.exp(lp)
    return float(np.mean(np.clip(total, 0.0, 1.0)))


def nway_mean_rank(ranks: ArrayLike, n_items: int, n_way: int) -> float:
    """Expected rank in an ``n_way`` sub-gallery (exact, by linearity)."""
    r = np.asarray(ranks, dtype=np.float64)
    m = float(n_items)
    n = float(n_way)
    if r.size == 0:
        return float("nan")
    if n >= m:
        return float(np.mean(r))
    return float(np.mean(1.0 + (n - 1.0) * (r - 1.0) / (m - 1.0)))


# --------------------------------------------------------------------------- #
# contract G entry point
# --------------------------------------------------------------------------- #
def retrieval_metrics(
    z_eeg: ArrayLike,
    z_vid_gallery: ArrayLike,
    true_idx: ArrayLike,
    groups: Optional[ArrayLike] = None,
    *,
    gallery_sizes: Sequence[int] = DEFAULT_GALLERY_SIZES,
    topk: Sequence[int] = (1, 5),
    include_cross_group: bool = True,
    assume_normalized: bool = True,
) -> Dict[str, float]:
    """Contract G. Retrieval metrics for one set of queries against one gallery.

    Parameters
    ----------
    z_eeg : (B, D) query embeddings (EEG side for ``eeg2vid``).
    z_vid_gallery : (M, D) gallery embeddings, one row per candidate item.
    true_idx : (B,) index into the gallery of the correct item per query.
    groups : optional (M,) integer/str group label per gallery item.  Normally
        the material id.  Enables ``within_group_*`` (attribute-shortcut
        ceiling) and ``cross_group_*`` keys.
    gallery_sizes : N-way gallery sizes; sizes ``>= M`` are reported as the full
        gallery and sizes ``> M`` are skipped.
    assume_normalized : if False, both sides are L2-normalised first.

    Returns
    -------
    Flat dict.  Guaranteed keys (contract G): ``top1``, ``top5``, ``mean_rank``
    and, when ``groups`` is given, ``within_group_top1``.  Additional keys:
    ``median_rank``, ``mrr``, ``n_queries``, ``n_items``, ``chance_top1``,
    ``chance_top5``, ``chance_mean_rank``, ``nway{N}_top1`` / ``_top5`` /
    ``_mean_rank`` / ``_chance_top1``, ``within_group_*``, ``cross_group_*``.
    """
    z_q = to_numpy(z_eeg).astype(np.float64, copy=False)
    z_g = to_numpy(z_vid_gallery).astype(np.float64, copy=False)
    true_idx = np.asarray(to_numpy(true_idx), dtype=np.int64).ravel()

    if z_q.ndim != 2 or z_g.ndim != 2:
        raise ValueError("z_eeg and z_vid_gallery must both be 2-D (B, D) / (M, D)")
    if z_q.shape[1] != z_g.shape[1]:
        raise ValueError(f"embedding dim mismatch: {z_q.shape[1]} vs {z_g.shape[1]}")
    if true_idx.shape[0] != z_q.shape[0]:
        raise ValueError("true_idx must have one entry per query")

    if not assume_normalized:
        z_q = l2_normalize(z_q)
        z_g = l2_normalize(z_g)

    n_q, n_items = z_q.shape[0], z_g.shape[0]
    out: Dict[str, float] = {"n_queries": float(n_q), "n_items": float(n_items)}
    if n_q == 0 or n_items < 2:
        return out

    scores = z_q @ z_g.T  # (B, M) cosine similarity
    ranks = rank_of_true(scores, true_idx)

    # ---- full gallery -----------------------------------------------------
    for k in topk:
        out[f"top{k}"] = float(np.mean(ranks <= k))
        out[f"chance_top{k}"] = float(min(k, n_items) / n_items)
    out["mean_rank"] = float(np.mean(ranks))
    out["median_rank"] = float(np.median(ranks))
    out["mrr"] = float(np.mean(1.0 / ranks))
    out["chance_mean_rank"] = float((n_items + 1) / 2.0)

    # ---- N-way sub-galleries (analytic) -----------------------------------
    for n_way in gallery_sizes:
        if n_way > n_items:
            continue
        tag = f"nway{n_way}"
        for k in topk:
            if k >= n_way:
                continue
            out[f"{tag}_top{k}"] = nway_topk_accuracy(ranks, n_items, n_way, k)
            out[f"{tag}_chance_top{k}"] = float(k / n_way)
        out[f"{tag}_mean_rank"] = nway_mean_rank(ranks, n_items, n_way)
        out[f"{tag}_chance_mean_rank"] = float((n_way + 1) / 2.0)

    # ---- group-restricted galleries ---------------------------------------
    if groups is None:
        return out

    g = np.asarray(to_numpy(groups)).ravel()
    if g.shape[0] != n_items:
        raise ValueError("groups must have one label per gallery item")
    g_codes = pd.Categorical(g).codes.astype(np.int64)
    q_group = g_codes[true_idx]

    within_ranks: List[np.ndarray] = []
    within_sizes: List[np.ndarray] = []
    cross_ranks: List[np.ndarray] = []
    cross_sizes: List[np.ndarray] = []
    for code in np.unique(q_group):
        q_sel = np.flatnonzero(q_group == code)
        same = g_codes == code
        n_same = int(same.sum())

        if n_same >= 2:
            sub_scores = scores[np.ix_(q_sel, np.flatnonzero(same))]
            remap = np.full(n_items, -1, dtype=np.int64)
            remap[np.flatnonzero(same)] = np.arange(n_same)
            within_ranks.append(rank_of_true(sub_scores, remap[true_idx[q_sel]]))
            within_sizes.append(np.full(q_sel.size, n_same, dtype=np.float64))

        if include_cross_group:
            other_cols = np.flatnonzero(~same)
            if other_cols.size >= 1:
                # build a per-query gallery of {other-group items} + {true item}
                sub = np.empty((q_sel.size, other_cols.size + 1), dtype=np.float64)
                sub[:, : other_cols.size] = scores[np.ix_(q_sel, other_cols)]
                sub[:, -1] = scores[q_sel, true_idx[q_sel]]
                true_col = np.full(q_sel.size, other_cols.size, dtype=np.int64)
                cross_ranks.append(rank_of_true(sub, true_col))
                cross_sizes.append(np.full(q_sel.size, other_cols.size + 1, dtype=np.float64))

    if within_ranks:
        wr = np.concatenate(within_ranks)
        ws = np.concatenate(within_sizes)
        for k in topk:
            out[f"within_group_top{k}"] = float(np.mean(wr <= k))
        out["within_group_mean_rank"] = float(np.mean(wr))
        out["within_group_size_mean"] = float(np.mean(ws))
        out["within_group_n_queries"] = float(wr.size)
        # chance is heterogeneous across group sizes -> average of 1/|group|
        out["within_group_chance_top1"] = float(np.mean(1.0 / ws))
        out["within_group_chance_top5"] = float(np.mean(np.minimum(5.0, ws) / ws))
        # N-way within group, harmonised across heterogeneous group sizes
        for n_way in gallery_sizes:
            usable = ws >= n_way
            if usable.sum() < 1 or n_way < 2:
                continue
            accs, mrs = [], []
            for size in np.unique(ws[usable]):
                sel = usable & (ws == size)
                accs.append(
                    (sel.sum(), nway_topk_accuracy(wr[sel], int(size), n_way, 1))
                )
                mrs.append((sel.sum(), nway_mean_rank(wr[sel], int(size), n_way)))
            wsum = float(sum(n for n, _ in accs))
            out[f"within_group_nway{n_way}_top1"] = float(
                sum(n * v for n, v in accs) / wsum
            )
            out[f"within_group_nway{n_way}_mean_rank"] = float(
                sum(n * v for n, v in mrs) / wsum
            )
            out[f"within_group_nway{n_way}_chance_top1"] = float(1.0 / n_way)
    if cross_ranks:
        cr = np.concatenate(cross_ranks)
        cs = np.concatenate(cross_sizes)
        for k in topk:
            out[f"cross_group_top{k}"] = float(np.mean(cr <= k))
        out["cross_group_mean_rank"] = float(np.mean(cr))
        out["cross_group_chance_top1"] = float(np.mean(1.0 / cs))
        out["cross_group_size_mean"] = float(np.mean(cs))

    return out


# --------------------------------------------------------------------------- #
# flat dict -> long dataframe
# --------------------------------------------------------------------------- #
_METRIC_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("full", re.compile(r"^(top\d+|mean_rank|median_rank|mrr)$")),
    ("nway", re.compile(r"^nway(\d+)_(top\d+|mean_rank)$")),
    ("within_group_full", re.compile(r"^within_group_(top\d+|mean_rank)$")),
    ("within_group_nway", re.compile(r"^within_group_nway(\d+)_(top\d+|mean_rank)$")),
    ("cross_group_full", re.compile(r"^cross_group_(top\d+|mean_rank)$")),
)


def metrics_to_long(metrics: Dict[str, float], **fixed: Any) -> pd.DataFrame:
    """Turn the flat dict from :func:`retrieval_metrics` into tidy long rows.

    Columns: ``gallery``, ``metric``, ``value``, ``chance`` plus ``n_queries``,
    ``n_items`` and any keyword arguments in ``fixed`` (e.g. ``subject_id``,
    ``direction``, ``trial_type``, ``fold``).
    """
    rows: List[Dict[str, Any]] = []
    n_q = metrics.get("n_queries", np.nan)
    n_items = metrics.get("n_items", np.nan)
    for key, value in metrics.items():
        if key.startswith("chance_") or "_chance_" in key:
            continue
        if key in ("n_queries", "n_items", "within_group_n_queries",
                   "within_group_size_mean", "cross_group_size_mean"):
            continue
        gallery: Optional[str] = None
        metric: Optional[str] = None
        for kind, pat in _METRIC_PATTERNS:
            m = pat.match(key)
            if not m:
                continue
            if kind == "full":
                gallery, metric = "full", m.group(1)
            elif kind == "nway":
                gallery, metric = f"nway{m.group(1)}", m.group(2)
            elif kind == "within_group_full":
                gallery, metric = "within_group_full", m.group(1)
            elif kind == "within_group_nway":
                gallery, metric = f"within_group_nway{m.group(1)}", m.group(2)
            elif kind == "cross_group_full":
                gallery, metric = "cross_group_full", m.group(1)
            break
        if gallery is None or metric is None:
            continue
        prefix = "" if gallery in ("full",) else ""
        chance_key = _chance_key_for(key)
        rows.append(
            {
                **fixed,
                "gallery": gallery,
                "metric": metric,
                "value": float(value),
                "chance": float(metrics.get(chance_key, np.nan)),
                "n_queries": float(n_q),
                "n_items": float(n_items),
                "_prefix": prefix,
            }
        )
    df = pd.DataFrame(rows)
    if "_prefix" in df.columns:
        df = df.drop(columns=["_prefix"])
    return df


def _chance_key_for(key: str) -> str:
    """Map a metric key to its chance-level key as emitted by retrieval_metrics."""
    if key.startswith("nway"):
        head, tail = key.split("_", 1)
        return f"{head}_chance_{tail}"
    if key.startswith("within_group_nway"):
        head, tail = key[len("within_group_") :].split("_", 1)
        return f"within_group_{head}_chance_{tail}"
    if key.startswith("within_group_"):
        return "within_group_chance_" + key[len("within_group_") :]
    if key.startswith("cross_group_"):
        return "cross_group_chance_" + key[len("cross_group_") :]
    return "chance_" + key


# --------------------------------------------------------------------------- #
# pseudo-trials
# --------------------------------------------------------------------------- #
def pseudo_trial_index(
    item_ids: ArrayLike,
    *,
    k: int = 4,
    subject_ids: Optional[ArrayLike] = None,
    rng: Optional[np.random.Generator] = None,
    allow_partial: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Draw one pseudo-trial per (subject, item) group.

    Returns
    -------
    idx : (G, k) int array of row indices to average (``-1`` padded if
        ``allow_partial`` and a group has fewer than ``k`` trials).
    out_items : (G,) item id per pseudo-trial.
    out_subjects : (G,) subject id per pseudo-trial (``-1`` if not supplied).
    n_dropped : number of (subject, item) groups dropped for having < k trials.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    items = np.asarray(to_numpy(item_ids)).ravel()
    subs = (
        np.asarray(to_numpy(subject_ids)).ravel()
        if subject_ids is not None
        else np.full(items.shape[0], -1)
    )
    if subs.shape[0] != items.shape[0]:
        raise ValueError("subject_ids must align with item_ids")

    order = np.lexsort((items, subs))
    items_s, subs_s = items[order], subs[order]
    boundaries = np.flatnonzero(
        np.r_[True, (items_s[1:] != items_s[:-1]) | (subs_s[1:] != subs_s[:-1])]
    )
    starts = boundaries
    ends = np.r_[boundaries[1:], items_s.shape[0]]

    idx_rows: List[np.ndarray] = []
    out_items: List[Any] = []
    out_subs: List[Any] = []
    n_dropped = 0
    for s, e in zip(starts, ends):
        members = order[s:e]
        if members.size < k:
            if not allow_partial:
                n_dropped += 1
                continue
            pad = np.full(k - members.size, -1, dtype=np.int64)
            chosen = np.concatenate([members, pad])
        else:
            chosen = rng.choice(members, size=k, replace=False)
        idx_rows.append(np.asarray(chosen, dtype=np.int64))
        out_items.append(items_s[s])
        out_subs.append(subs_s[s])

    if not idx_rows:
        return (
            np.zeros((0, k), dtype=np.int64),
            np.zeros(0, dtype=items.dtype),
            np.zeros(0, dtype=subs.dtype),
            n_dropped,
        )
    return (
        np.stack(idx_rows, axis=0),
        np.asarray(out_items),
        np.asarray(out_subs),
        n_dropped,
    )


def build_pseudo_trials(
    z: ArrayLike,
    item_ids: ArrayLike,
    *,
    k: int = 4,
    subject_ids: Optional[ArrayLike] = None,
    rng: Optional[np.random.Generator] = None,
    renormalize: bool = True,
    disjoint_halves: bool = False,
    k_second: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average ``k`` trials of the same (subject, item) into one pseudo-trial.

    With ``disjoint_halves=True`` two *non-overlapping* pseudo-trials are drawn
    per group (requires ``k + k_second`` trials per group, default ``2k``) and
    the function returns ``(z_a, z_b, items, subjects)``.  That variant is what
    :mod:`tactus.eval.noise_ceiling` uses for split-half reliability and for the
    retrieval ceiling; ``k_second`` lets the two halves be asymmetric, e.g.
    ``k=1, k_second=7`` for "single-trial query against the cleanest available
    gallery".
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    z_arr = to_numpy(z).astype(np.float64, copy=False)
    if disjoint_halves:
        k2 = int(k_second if k_second is not None else k)
        idx, items, subs, _ = pseudo_trial_index(
            item_ids, k=k + k2, subject_ids=subject_ids, rng=rng, allow_partial=False
        )
        if idx.shape[0] == 0:
            empty = np.zeros((0, z_arr.shape[1]))
            return empty, empty, np.zeros(0), np.zeros(0)
        z_a = z_arr[idx[:, :k]].mean(axis=1)
        z_b = z_arr[idx[:, k:]].mean(axis=1)
        if renormalize:
            z_a, z_b = l2_normalize(z_a), l2_normalize(z_b)
        return z_a, z_b, items, subs

    idx, items, subs, _ = pseudo_trial_index(
        item_ids, k=k, subject_ids=subject_ids, rng=rng, allow_partial=False
    )
    if idx.shape[0] == 0:
        return np.zeros((0, z_arr.shape[1])), np.zeros(0), np.zeros(0)
    z_p = z_arr[idx].mean(axis=1)
    if renormalize:
        z_p = l2_normalize(z_p)
    return z_p, items, subs


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalConfig:
    """Configuration for the full retrieval endpoint zoo."""

    gallery_sizes: Tuple[int, ...] = DEFAULT_GALLERY_SIZES
    topk: Tuple[int, ...] = (1, 5)
    pseudo_k: int = 4
    n_pseudo_resamples: int = 20
    directions: Tuple[str, ...] = ("eeg2vid", "vid2eeg")
    per_subject: bool = True
    pooled: bool = True
    include_cross_group: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        bad = set(self.directions) - {"eeg2vid", "vid2eeg"}
        if bad:
            raise ValueError(f"unknown retrieval directions: {sorted(bad)}")


def _mean_metric_dicts(dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of flat metric dicts key-wise (missing keys ignored)."""
    if not dicts:
        return {}
    keys = sorted(set().union(*[set(d) for d in dicts]))
    out: Dict[str, float] = {}
    for key in keys:
        vals = [d[key] for d in dicts if key in d and np.isfinite(d[key])]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def evaluate_retrieval(
    z_eeg: ArrayLike,
    trial_item_id: ArrayLike,
    gallery_emb: ArrayLike,
    gallery_item_ids: ArrayLike,
    *,
    groups: Optional[ArrayLike] = None,
    subject_ids: Optional[ArrayLike] = None,
    cfg: Optional[RetrievalConfig] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run the whole retrieval endpoint zoo and return tidy long rows.

    Parameters
    ----------
    z_eeg : (B, D) L2-normalised EEG embeddings of held-out trials.
    trial_item_id : (B,) item id per trial.  Use ``condition_id`` for the
        360-way gallery and ``video_id`` for the 90-way / 18-way base-video
        gallery.  The pseudo-trial grouping follows this same id, so passing
        ``video_id`` averages across orientations as well as repeats.
    gallery_emb : (M, D) L2-normalised projected video embeddings.
    gallery_item_ids : (M,) item ids matching ``trial_item_id``'s space.
    groups : optional (M,) group label per gallery item (material id).
    subject_ids : optional (B,) subject id per trial.  Required for
        ``cfg.per_subject`` and for correct pseudo-trial construction.
    extra_fields : constant columns added to every row (fold id, regime, ...).

    Returns
    -------
    Long DataFrame with columns ``subject_id, direction, trial_type, gallery,
    metric, value, chance, n_queries, n_items`` (+ ``extra_fields``).
    """
    cfg = cfg or RetrievalConfig()
    extra_fields = dict(extra_fields or {})
    rng = np.random.default_rng(cfg.seed)

    z_q = l2_normalize(to_numpy(z_eeg))
    g_emb = l2_normalize(to_numpy(gallery_emb))
    g_ids = np.asarray(to_numpy(gallery_item_ids)).ravel()
    t_ids = np.asarray(to_numpy(trial_item_id)).ravel()
    subs = (
        np.asarray(to_numpy(subject_ids)).ravel()
        if subject_ids is not None
        else np.full(t_ids.shape[0], -1)
    )
    grp = np.asarray(to_numpy(groups)).ravel() if groups is not None else None

    order = np.argsort(g_ids)
    g_ids_sorted = g_ids[order]
    pos = np.searchsorted(g_ids_sorted, t_ids)
    if np.any(pos >= g_ids_sorted.shape[0]) or np.any(g_ids_sorted[np.minimum(pos, g_ids_sorted.shape[0] - 1)] != t_ids):
        raise ValueError("every trial_item_id must exist in gallery_item_ids")
    true_idx = order[pos]

    frames: List[pd.DataFrame] = []
    subject_sets: List[Tuple[Any, np.ndarray]] = []
    if cfg.pooled:
        subject_sets.append(("pooled", np.arange(t_ids.shape[0])))
    if cfg.per_subject and subject_ids is not None:
        for s in np.unique(subs):
            subject_sets.append((s, np.flatnonzero(subs == s)))

    for subject_key, sel in subject_sets:
        if sel.size == 0:
            continue
        z_sel, true_sel, item_sel, sub_sel = z_q[sel], true_idx[sel], t_ids[sel], subs[sel]
        base = dict(extra_fields, subject_id=subject_key)

        # ---------------- eeg -> video ------------------------------------
        if "eeg2vid" in cfg.directions:
            m = retrieval_metrics(
                z_sel, g_emb, true_sel, grp,
                gallery_sizes=cfg.gallery_sizes, topk=cfg.topk,
                include_cross_group=cfg.include_cross_group,
            )
            frames.append(metrics_to_long(m, **base, direction="eeg2vid", trial_type="single"))

            per_resample: List[Dict[str, float]] = []
            for _ in range(cfg.n_pseudo_resamples):
                z_p, items_p, _ = build_pseudo_trials(
                    z_sel, item_sel, k=cfg.pseudo_k,
                    subject_ids=sub_sel, rng=rng,
                )
                if z_p.shape[0] == 0:
                    continue
                pos_p = np.searchsorted(g_ids_sorted, items_p)
                true_p = order[pos_p]
                per_resample.append(
                    retrieval_metrics(
                        z_p, g_emb, true_p, grp,
                        gallery_sizes=cfg.gallery_sizes, topk=cfg.topk,
                        include_cross_group=cfg.include_cross_group,
                    )
                )
            if per_resample:
                frames.append(
                    metrics_to_long(
                        _mean_metric_dicts(per_resample),
                        **base, direction="eeg2vid",
                        trial_type=f"pseudo{cfg.pseudo_k}",
                    )
                )

        # ---------------- video -> eeg ------------------------------------
        # Gallery = one EEG representation per candidate item; query = that
        # item's video embedding.  Sensitive to hubness / collapsed EEG space.
        if "vid2eeg" in cfg.directions:
            for trial_type, k_use in (("single", 1), (f"pseudo{cfg.pseudo_k}", cfg.pseudo_k)):
                per_resample = []
                for _ in range(cfg.n_pseudo_resamples):
                    z_p, items_p, _ = build_pseudo_trials(
                        z_sel, item_sel, k=k_use, subject_ids=sub_sel, rng=rng
                    )
                    if z_p.shape[0] < 2:
                        continue
                    # one EEG gallery row per item; in pooled mode the same item
                    # appears once per subject, so average them rather than
                    # arbitrarily keeping whichever subject sorted first
                    uniq_items, inv = np.unique(items_p, return_inverse=True)
                    z_gal_eeg = np.zeros((uniq_items.size, z_p.shape[1]))
                    np.add.at(z_gal_eeg, inv, z_p)
                    counts = np.bincount(inv, minlength=uniq_items.size).astype(np.float64)
                    z_gal_eeg = l2_normalize(z_gal_eeg / counts[:, None])
                    pos_u = np.searchsorted(g_ids_sorted, uniq_items)
                    q_vid = g_emb[order[pos_u]]
                    g_sub = None
                    if grp is not None:
                        g_sub = grp[order[pos_u]]
                    per_resample.append(
                        retrieval_metrics(
                            q_vid, z_gal_eeg, np.arange(uniq_items.shape[0]), g_sub,
                            gallery_sizes=cfg.gallery_sizes, topk=cfg.topk,
                            include_cross_group=cfg.include_cross_group,
                        )
                    )
                if per_resample:
                    frames.append(
                        metrics_to_long(
                            _mean_metric_dicts(per_resample),
                            **base, direction="vid2eeg", trial_type=trial_type,
                        )
                    )

    if not frames:
        return pd.DataFrame(
            columns=["subject_id", "direction", "trial_type", "gallery",
                     "metric", "value", "chance", "n_queries", "n_items"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out


def primary_endpoint(df: pd.DataFrame, *, pseudo_k: int = 4) -> pd.DataFrame:
    """Filter a long retrieval frame down to the pre-registered primary endpoint.

    Primary endpoint (BLUEPRINT v2 §5.2): within-subject, video-disjoint,
    18-way top-1, pseudo-trial k=4, eeg->video, aggregated over video folds.
    """
    return df[
        (df["direction"] == "eeg2vid")
        & (df["trial_type"] == f"pseudo{pseudo_k}")
        & (df["gallery"] == "nway18")
        & (df["metric"] == "top1")
    ].copy()
