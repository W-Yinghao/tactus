"""Factorization probes for the FHMC arm (DECISIONS D20).

The flagship objective claims a *factorized* embedding: a content subspace that
carries what is being touched and is blind to how it is oriented, a geometry
subspace that carries orientation and little else, a semantic subspace tracking
affect, and neither of them carrying subject identity.  That claim is not
testable from the retrieval endpoint -- the endpoint only sees the trunk.  This
module reads the trunk embeddings a fold already wrote and the head weights the
checkpoint already stores, rebuilds each subspace offline, and decodes the
attributes out of it.  No retraining, no GPU.

Two probe designs, deliberately different:

* **stimulus attributes** (video, orientation, material, touch_type) are probed
  with *subject-grouped* CV: train on some subjects, test on others.  A
  subspace that only carries the attribute in a subject-specific code scores at
  chance here, which is the property we actually want to claim.
* **subject identity** is probed with *video-grouped* CV: train on some held-out
  videos, test on others.  Grouping the other way would let the probe recover
  the subject from whichever video happened to be in the training split.

Both probe per-(subject, condition) trial averages rather than single trials,
matching the pseudo-trial unit of the primary endpoint.

Read the material row against DECISIONS D17: on these 90 stimuli material,
toucher and object are collinear at r ~ 1.0, so "material decodes from the
content head" and "video identity decodes from the content head" are close to
the same statement, not two pieces of evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Subspaces are rebuilt from these checkpoint tensors; the names must track
# FactorizedFHMC's attribute names.
HEADS = {"content": "head_content.weight",
         "geometry": "head_geometry.weight",
         "semantic": "head_semantic.weight"}

#: Grouping variable per probe target.  See the module docstring for why these
#: differ -- getting them the wrong way round inflates both rows.
TARGETS = [("video_id", "subject_id"),
           ("orientation", "subject_id"),
           ("material", "subject_id"),
           ("touch_type", "subject_id"),
           ("subject_id", "video_id")]


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def subspaces(z: np.ndarray, loss_sd: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Trunk plus one array per factor head, all L2-normalized as in training."""
    out = {"trunk": _l2(np.asarray(z, dtype=np.float64))}
    for name, key in HEADS.items():
        if key not in loss_sd:
            continue
        W = np.asarray(loss_sd[key], dtype=np.float64)   # (d_head, dim), no bias
        out[name] = _l2(z @ W.T)
    return out


def pseudo_average(emb: Dict[str, np.ndarray], meta: pd.DataFrame):
    """Collapse trials to one row per (subject, condition), renormalized.

    The endpoint scores pseudo-trials, so probing single trials would measure a
    different (noisier) object than the number the probes are meant to explain.
    """
    key = meta.groupby(["subject_id", "condition_id"], sort=True).ngroup().to_numpy()
    order = np.argsort(key, kind="stable")
    k_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    counts = np.diff(np.r_[starts, k_sorted.size]).astype(np.float64)
    rows = meta.iloc[order[starts]].reset_index(drop=True)
    avg = {}
    for name, x in emb.items():
        s = np.add.reduceat(x[order], starts, axis=0)
        avg[name] = _l2(s / counts[:, None])
    return avg, rows, counts


def probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0,
          max_train: int = 20000) -> Dict[str, float]:
    """Grouped-CV linear decode; accuracy alongside its own chance level."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    classes, y_enc = np.unique(y, return_inverse=True)
    if classes.size < 2:
        return {"acc": float("nan"), "chance": float("nan"), "n_classes": int(classes.size),
                "n": int(y.size), "above_chance": False, "majority_rate": float("nan")}
    n_splits = min(5, np.unique(groups).size)
    rng = np.random.default_rng(seed)
    accs: List[float] = []
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y_enc, groups):
        # A class absent from train cannot be predicted; keep test comparable.
        keep = np.isin(y_enc[te], np.unique(y_enc[tr]))
        te = te[keep]
        if te.size == 0:
            continue
        if tr.size > max_train:
            tr = rng.choice(tr, size=max_train, replace=False)
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf.fit(sc.transform(x[tr]), y_enc[tr])
        accs.append(float((clf.predict(sc.transform(x[te])) == y_enc[te]).mean()))
    if not accs:
        return {"acc": float("nan"), "chance": float("nan"), "n_classes": int(classes.size),
                "n": int(y.size), "above_chance": False, "majority_rate": float("nan")}
    counts = np.bincount(y_enc)
    return {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
            "chance": 1.0 / classes.size,
            # Reported next to acc because an imbalanced target makes 1/n_classes
            # the wrong yardstick -- the same trap that inflated MVPA material.
            "majority_rate": float(counts.max() / counts.sum()),
            "n_classes": int(classes.size), "n": int(y.size),
            "above_chance": bool(np.mean(accs) > max(1.0 / classes.size,
                                                     counts.max() / counts.sum()))}


def alignment_retention(avg: Dict[str, np.ndarray], rows: pd.DataFrame,
                        gallery_ids: np.ndarray, z_vid: np.ndarray,
                        loss_sd: Dict[str, Any]) -> pd.DataFrame:
    """18-way top-1 inside each subspace, gallery projected through the same head.

    The heads are shared across modalities, so the video gallery goes through
    the identical linear map.  A subspace that decodes attributes but retains no
    retrieval is carrying the attribute in a direction the endpoint never uses.
    """
    gal = {"trunk": _l2(np.asarray(z_vid, dtype=np.float64))}
    for name, key in HEADS.items():
        if key in loss_sd:
            gal[name] = _l2(np.asarray(z_vid, dtype=np.float64) @ np.asarray(loss_sd[key], dtype=np.float64).T)
    truth = np.searchsorted(gallery_ids, rows["video_id"].to_numpy())
    out = []
    for name, g in gal.items():
        pred = (avg[name] @ g.T).argmax(axis=1)
        out.append({"subspace": name, "top1": float((pred == truth).mean()),
                    "chance": 1.0 / gallery_ids.size, "n_queries": int(truth.size)})
    return pd.DataFrame(out)


def run_fold(fold: Path, cond: pd.DataFrame, seed: int = 0) -> Dict[str, pd.DataFrame]:
    import torch

    d = np.load(fold / "test_embeddings.npz", allow_pickle=True)
    ck = torch.load(fold / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    loss_sd = {k: v.numpy() for k, v in ck.get("loss", {}).items() if hasattr(v, "numpy")}
    if not any(k in loss_sd for k in HEADS.values()):
        raise RuntimeError(f"{fold}: checkpoint has no factor heads -- not an FHMC run")

    meta = pd.DataFrame({"subject_id": d["subject_id"], "condition_id": d["condition_id"],
                         "video_id": d["video_id"]}).merge(cond, on="condition_id",
                                                           how="left", suffixes=("", "_c"))
    emb = subspaces(np.asarray(d["z_eeg"], dtype=np.float64), loss_sd)
    avg, rows, counts = pseudo_average(emb, meta)

    recs: List[Dict[str, Any]] = []
    for target, group in TARGETS:
        y = rows[target].to_numpy()
        ok = pd.notna(y)
        for name, x in avg.items():
            r = probe(x[ok], y[ok], rows[group].to_numpy()[ok], seed=seed)
            recs.append({"fold": fold.name, "subspace": name, "target": target,
                         "grouped_by": group, **r})
    ret = alignment_retention(avg, rows, np.asarray(d["gallery_video_ids"]),
                              np.asarray(d["z_vid_video"]), loss_sd)
    ret.insert(0, "fold", fold.name)
    return {"probes": pd.DataFrame(recs), "retention": ret,
            "k": pd.DataFrame([{"fold": fold.name, "mean_trials_per_pseudo": float(counts.mean()),
                                "n_pseudo": int(counts.size)}])}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tactus.eval.factor_probes")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--trials", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", default=None,
                    help="comma-separated fold directory names to keep, e.g. "
                         "'vf00,vf01,vf02,vf03'. Use it to exclude the sealed "
                         "confirmation fold when probing for model selection "
                         "(DECISIONS D16), and to compare two arms over the same "
                         "folds when one has run fewer of them.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from tactus.eval.run_report import fold_dirs

    cond = (pd.read_parquet(a.trials)
              .drop_duplicates("condition_id")
              [["condition_id", "orientation", "material", "touch_type", "toucher"]]
              .reset_index(drop=True))
    folds = fold_dirs(a.run_dir, a.regime)
    if a.folds:
        keep = {x.strip() for x in a.folds.split(",") if x.strip()}
        missing = keep - {f.name for f in folds}
        if missing:
            print(f"[probes] requested fold(s) not present: {sorted(missing)}",
                  file=sys.stderr)
            return 1
        folds = [f for f in folds if f.name in keep]
    if not folds:
        print(f"[probes] no folds for regime={a.regime!r} under {a.run_dir}", file=sys.stderr)
        return 1

    parts: Dict[str, List[pd.DataFrame]] = {"probes": [], "retention": [], "k": []}
    for fd in folds:
        print(f"[probes] {fd.name} ...", file=sys.stderr)
        for key, df in run_fold(fd, cond, seed=a.seed).items():
            parts[key].append(df)

    a.out.mkdir(parents=True, exist_ok=True)
    probes = pd.concat(parts["probes"], ignore_index=True)
    reten = pd.concat(parts["retention"], ignore_index=True)
    probes.to_csv(a.out / "probes_per_fold.csv", index=False)
    reten.to_csv(a.out / "retention_per_fold.csv", index=False)

    agg = (probes.groupby(["subspace", "target"], as_index=False)
                 .agg(acc=("acc", "mean"), acc_sd=("acc", "std"), chance=("chance", "mean"),
                      majority_rate=("majority_rate", "mean"), n_classes=("n_classes", "max"),
                      n_folds=("acc", "size")))
    agg["above"] = agg["acc"] - np.maximum(agg["chance"], agg["majority_rate"])
    ragg = reten.groupby("subspace", as_index=False).agg(top1=("top1", "mean"),
                                                         top1_sd=("top1", "std"),
                                                         chance=("chance", "mean"))
    agg.to_csv(a.out / "probes.csv", index=False)
    ragg.to_csv(a.out / "retention.csv", index=False)
    (a.out / "PROBES.md").write_text(
        "# FHMC factorization probes (DECISIONS D20)\n\n"
        f"- run: `{a.run_dir}`  regime: `{a.regime}`  folds: {len(folds)}\n"
        f"- probe unit: per-(subject, condition) trial average "
        f"(mean {pd.concat(parts['k'])['mean_trials_per_pseudo'].mean():.2f} trials)\n"
        "- stimulus attributes are subject-grouped CV; subject identity is video-grouped CV\n"
        "- `above` = acc - max(chance, majority_rate); an imbalanced target makes\n"
        "  1/n_classes the wrong yardstick on its own\n\n"
        "## Attribute decoding per subspace\n\n" + agg.to_markdown(index=False) +
        "\n\n## Alignment retention (18-way top-1 inside each subspace)\n\n" +
        ragg.to_markdown(index=False) +
        "\n\nMaterial must be read against D17: material, toucher and object are\n"
        "collinear at r ~ 1.0 across these 90 stimuli, so the material and video\n"
        "rows are close to one statement rather than two.\n")
    print(agg.to_string(index=False))
    print()
    print(ragg.to_string(index=False))
    print(f"[written] {a.out / 'PROBES.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
