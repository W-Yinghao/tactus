#!/usr/bin/env python
"""Phase-0 video-encoder census — gate G4 (BLUEPRINT_v2 section 4.1).

Consumes the frozen embedding caches written by
:mod:`tactus.models.video.encode` (``{tag}.npz`` with ``cond_emb (360, D)`` and
``base_emb (90, D)``) and renders the three **pre-registered judgments** that
decide the shape of the primary endpoint:

1. **Collapse.**  Mean pairwise cosine over the 90 base videos and the PC
   variance spectrum.  Pre-registered rule: collapsed iff mean cosine ``> 0.9``
   **and** top-3 PC variance ``> 0.8``.  Both, not either -- a high mean cosine
   alone is the normal signature of an encoder with a large constant component
   and says nothing about usable rank.

2. **RDM alignment with behaviour.**  Spearman between the embedding RDM and
   each attribute RDM, with a **video-level permutation null** (the exchangeable
   unit is the source video, never the trial -- blueprint 5.2).  Go condition:
   at least one attribute RDM significantly correlated.

3. **Leave-one-video-out attribute predictability.**  Ridge from VTD attributes
   to the held-out video's embedding, retrieved by cosine among all 90.
   Pre-registered rule: **top-5 not above the permutation null => video-level
   zero-shot is impossible in principle, and the primary endpoint changes to
   attribute-level generalisation.**  That is a contingency, not a failure.

The report also carries the honest caveat that judgment 2 and 3 are evaluated on
90 stimuli whose attributes are severely collinear (audit C found 23 attribute
pairs above 0.5, three of them at ~1.0), so a "significant" attribute RDM
correlation identifies a *bundle*, not an isolated attribute.

CLI
---
    python -m tactus.eval.census --emb-dir $TACTUS_WORK/derived/video_emb \
        --vtd $TACTUS_BIDS/code/analysis/VTD.csv --out $TACTUS_WORK/phase0_out
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..data.events import load_vtd
from .rsa import (
    rdm_from_categorical,
    rdm_from_continuous,
    rdm_from_embeddings,
    spearman_rdm_corr,
    vectorize_rdm,
)

N_VIDEOS = 90
CONT_COLS = ("valence", "arousal", "threat")
CAT_COLS = ("material", "touch_type", "toucher", "object", "approaching")


# --------------------------------------------------------------------------- #
# 1. collapse
# --------------------------------------------------------------------------- #
def collapse_diagnostics(base_emb: np.ndarray) -> Dict[str, object]:
    """Mean pairwise cosine + PC spectrum, with the pre-registered verdict."""
    e = base_emb / np.linalg.norm(base_emb, axis=1, keepdims=True)
    cos = e @ e.T
    iu = np.triu_indices(len(e), k=1)
    pair = cos[iu]

    centred = base_emb - base_emb.mean(0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    var = sv ** 2
    var = var / var.sum()
    top3, top10 = float(var[:3].sum()), float(var[:10].sum())

    mean_cos = float(pair.mean())
    collapsed = bool(mean_cos > 0.9 and top3 > 0.8)
    return {
        "mean_pairwise_cosine": round(mean_cos, 4),
        "min_pairwise_cosine": round(float(pair.min()), 4),
        "max_pairwise_cosine": round(float(pair.max()), 4),
        "pc_var_top3": round(top3, 4),
        "pc_var_top10": round(top10, 4),
        "effective_rank": round(float(np.exp(-(var * np.log(var + 1e-12)).sum())), 2),
        "collapsed": collapsed,
        "rule": "collapsed iff mean cosine > 0.9 AND top-3 PC variance > 0.8",
    }


# --------------------------------------------------------------------------- #
# 2. RDM alignment, video-level permutation
# --------------------------------------------------------------------------- #
def attribute_rdms(vtd: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in CONT_COLS:
        if c in vtd.columns:
            out[c] = rdm_from_continuous(vtd[c].to_numpy())
    for c in CAT_COLS:
        if c in vtd.columns:
            out[c] = rdm_from_categorical(vtd[c].to_numpy())
    if "pain" in vtd.columns:
        out["pain"] = rdm_from_categorical(vtd["pain"].to_numpy())
    return out


def rdm_alignment(
    base_emb: np.ndarray, vtd: pd.DataFrame, *, n_perm: int = 5000, seed: int = 0
) -> List[Dict[str, object]]:
    """Spearman(embedding RDM, attribute RDM) with a video-level permutation null.

    The null permutes the 90 **video labels** of the attribute RDM (relabelling
    rows and columns together), which is the only exchangeable unit here.  A
    naive permutation of the RDM's off-diagonal entries would destroy the
    dependence structure among the 4005 pairs and give a null 4-9x too narrow.
    """
    emb_rdm = rdm_from_embeddings(base_emb, metric="cosine")
    emb_vec = vectorize_rdm(emb_rdm)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []

    for name, rdm in attribute_rdms(vtd).items():
        obs = spearman_rdm_corr(emb_vec, vectorize_rdm(rdm))
        null = np.empty(n_perm)
        for i in range(n_perm):
            p = rng.permutation(len(rdm))
            null[i] = spearman_rdm_corr(emb_vec, vectorize_rdm(rdm[np.ix_(p, p)]))
        # two-sided, +1 correction (Phipson & Smyth): p is never exactly 0
        pval = (1 + int(np.sum(np.abs(null) >= abs(obs)))) / (n_perm + 1)
        rows.append({
            "attribute": name,
            "spearman": round(float(obs), 4),
            "p_video_perm": round(float(pval), 5),
            "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std()), 4),
            "z": round(float((obs - null.mean()) / (null.std() + 1e-12)), 2),
        })
    rows.sort(key=lambda r: -abs(r["spearman"]))
    return rows


# --------------------------------------------------------------------------- #
# 3. LOVO attribute -> embedding predictability
# --------------------------------------------------------------------------- #
def _design_matrix(vtd: pd.DataFrame) -> np.ndarray:
    """VTD attributes as a numeric design: z-scored continuous + one-hot categorical."""
    blocks: List[np.ndarray] = []
    for c in CONT_COLS:
        if c in vtd.columns:
            v = vtd[c].to_numpy(dtype=float)
            blocks.append(((v - np.nanmean(v)) / (np.nanstd(v) + 1e-12))[:, None])
    for c in list(CAT_COLS) + ["pain"]:
        if c in vtd.columns:
            blocks.append(pd.get_dummies(vtd[c].astype(str), prefix=c).to_numpy(dtype=float))
    X = np.hstack(blocks)
    return np.nan_to_num(X)


def _ridge_fit_predict(X_tr: np.ndarray, Y_tr: np.ndarray, x_te: np.ndarray,
                       alpha: float) -> np.ndarray:
    xm, ym = X_tr.mean(0, keepdims=True), Y_tr.mean(0, keepdims=True)
    Xc, Yc = X_tr - xm, Y_tr - ym
    n_feat = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(n_feat), Xc.T @ Yc)
    return (x_te - xm) @ W + ym


def lovo_predictability(
    base_emb: np.ndarray, vtd: pd.DataFrame, *, alpha: float = 10.0,
    n_perm: int = 1000, seed: int = 0,
) -> Dict[str, object]:
    """Leave-one-video-out ridge from attributes to embedding, scored by retrieval."""
    X = _design_matrix(vtd)
    Y = base_emb / np.linalg.norm(base_emb, axis=1, keepdims=True)
    n = len(Y)

    def run(Xm: np.ndarray) -> Dict[str, float]:
        ranks = np.empty(n, dtype=int)
        for i in range(n):
            tr = np.arange(n) != i
            pred = _ridge_fit_predict(Xm[tr], Y[tr], Xm[i:i + 1], alpha)
            pred = pred / (np.linalg.norm(pred) + 1e-12)
            sims = (Y @ pred.T).ravel()
            ranks[i] = int(np.sum(sims > sims[i]))  # 0 = perfect
        return {
            "top1": float(np.mean(ranks == 0)),
            "top5": float(np.mean(ranks < 5)),
            "mean_rank": float(ranks.mean() + 1),
            "median_rank": float(np.median(ranks) + 1),
        }

    obs = run(X)
    rng = np.random.default_rng(seed)
    null = {k: np.empty(n_perm) for k in obs}
    for j in range(n_perm):
        # permute the video labels of the attribute matrix: the exchangeable unit
        Xp = X[rng.permutation(n)]
        r = run(Xp)
        for k in obs:
            null[k][j] = r[k]

    out: Dict[str, object] = {"alpha": alpha, "n_perm": n_perm, "chance_top1": 1 / n,
                              "chance_top5": 5 / n}
    for k, v in obs.items():
        nk = null[k]
        higher_is_better = k in ("top1", "top5")
        if higher_is_better:
            p = (1 + int(np.sum(nk >= v))) / (n_perm + 1)
        else:
            p = (1 + int(np.sum(nk <= v))) / (n_perm + 1)
        out[k] = round(float(v), 4)
        out[f"{k}_p"] = round(float(p), 5)
        out[f"{k}_null_mean"] = round(float(nk.mean()), 4)
    out["video_level_zeroshot_supported"] = bool(out["top5_p"] < 0.05)
    out["rule"] = ("top-5 not above the attribute-permutation null => video-level "
                   "zero-shot is impossible in principle; primary endpoint becomes "
                   "attribute-level generalisation (BLUEPRINT_v2 4.1)")
    return out


# --------------------------------------------------------------------------- #
# orientation structure (free, and Q1a depends on it)
# --------------------------------------------------------------------------- #
def orientation_structure(cond_emb: np.ndarray) -> Dict[str, object]:
    """How much of the encoder's geometry is orientation vs content?

    ``cond_emb`` is indexed ``condition_id = (video_id - 1) * 4 + orientation``.
    """
    e = cond_emb / np.linalg.norm(cond_emb, axis=1, keepdims=True)
    e = e.reshape(N_VIDEOS, 4, -1)
    within = [float((e[:, a] * e[:, b]).sum(-1).mean())
              for a in range(4) for b in range(a + 1, 4)]
    flat = e.reshape(N_VIDEOS * 4, -1)
    across = flat @ flat.T
    mask = np.ones_like(across, dtype=bool)
    vid = np.repeat(np.arange(N_VIDEOS), 4)
    mask &= vid[:, None] != vid[None, :]
    return {
        "within_video_across_orientation_cosine": round(float(np.mean(within)), 4),
        "across_video_cosine": round(float(across[mask].mean()), 4),
        "orientation_separability": round(
            float(np.mean(within) - across[mask].mean()), 4),
        "note": "orientation moves the embedding far less than content does when "
                "this gap is large; Q1a asks whether EEG shows the same ordering",
    }


# --------------------------------------------------------------------------- #
def census_one(npz_path: Path, vtd: pd.DataFrame, *, n_perm_rdm: int,
               n_perm_lovo: int, seed: int) -> Dict[str, object]:
    z = np.load(npz_path, allow_pickle=True)
    base = np.asarray(z["base_emb"], dtype=np.float64)
    cond = np.asarray(z["cond_emb"], dtype=np.float64)
    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {"raw": str(z["meta"])[:200]}
    return {
        "tag": npz_path.stem,
        "dim": int(base.shape[1]),
        "model_meta": meta,
        "collapse": collapse_diagnostics(base),
        "rdm_alignment": rdm_alignment(base, vtd, n_perm=n_perm_rdm, seed=seed),
        "lovo": lovo_predictability(base, vtd, n_perm=n_perm_lovo, seed=seed),
        "orientation": orientation_structure(cond),
    }


def render_report(results: List[Dict[str, object]]) -> str:
    L: List[str] = [
        "# TACTUS Phase-0 video-encoder census (gate G4)",
        "",
        f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Three pre-registered judgments (BLUEPRINT_v2 section 4.1). Permutation nulls",
        "use the **source video** as the exchangeable unit throughout.",
        "",
        "## 1. Embedding collapse",
        "",
        "| encoder | dim | mean cos | PC top3 | PC top10 | eff. rank | COLLAPSED |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r["collapse"]
        L.append(f"| {r['tag']} | {r['dim']} | {c['mean_pairwise_cosine']} | "
                 f"{c['pc_var_top3']} | {c['pc_var_top10']} | {c['effective_rank']} | "
                 f"{'**YES**' if c['collapsed'] else 'no'} |")
    L += ["", "Rule: collapsed iff mean cosine > 0.9 **and** top-3 PC variance > 0.8.", ""]

    L += ["## 2. Embedding RDM vs behavioural attribute RDMs", "",
          "| encoder | attribute | Spearman | z | p (video perm) |", "|---|---|---|---|---|"]
    for r in results:
        for row in r["rdm_alignment"][:6]:
            star = " **" if row["p_video_perm"] < 0.05 else " "
            L.append(f"| {r['tag']} | {row['attribute']} | {row['spearman']}{star}| "
                     f"{row['z']} | {row['p_video_perm']} |")
    L += ["", "**Caveat that must travel with every row above.** Audit C found 23 attribute",
          "pairs associated above 0.5 on these 90 stimuli, three of them at ~1.0",
          "(`toucher`-`object`-`material`). A significant row therefore identifies an",
          "attribute *bundle*, not an isolated attribute.", ""]

    L += ["## 3. Leave-one-video-out: do attributes predict a held-out embedding?", "",
          "| encoder | top-1 | top-5 | mean rank | p(top5) | video-level zero-shot |",
          "|---|---|---|---|---|---|"]
    for r in results:
        v = r["lovo"]
        L.append(f"| {r['tag']} | {v['top1']} | {v['top5']} | {v['mean_rank']} | "
                 f"{v['top5_p']} | {'supported' if v['video_level_zeroshot_supported'] else '**NOT supported**'} |")
    L += ["", f"Chance: top-1 = {1/N_VIDEOS:.4f}, top-5 = {5/N_VIDEOS:.4f}.",
          "If no encoder supports video-level zero-shot, the primary endpoint becomes",
          "attribute-level generalisation -- a pre-registered contingency, not a failure.", ""]

    L += ["## 4. Orientation geometry (Q1a prerequisite)", "",
          "| encoder | within-video across-orientation cos | across-video cos | gap |",
          "|---|---|---|---|"]
    for r in results:
        o = r["orientation"]
        L.append(f"| {r['tag']} | {o['within_video_across_orientation_cosine']} | "
                 f"{o['across_video_cosine']} | {o['orientation_separability']} |")

    L += ["", "## Gate G4 verdict", ""]
    any_align = any(any(row["p_video_perm"] < 0.05 for row in r["rdm_alignment"])
                    for r in results)
    any_lovo = any(r["lovo"]["video_level_zeroshot_supported"] for r in results)
    all_collapsed = all(r["collapse"]["collapsed"] for r in results)
    L.append(f"- at least one encoder x attribute RDM significant: **{any_align}**")
    L.append(f"- at least one encoder supports video-level zero-shot (LOVO): **{any_lovo}**")
    L.append(f"- every encoder collapsed: **{all_collapsed}**")
    if any_align and any_lovo and not all_collapsed:
        L.append("\n**G4 PASS** — primary endpoint stays video-level zero-shot retrieval.")
    elif any_align and not any_lovo:
        L.append("\n**G4 CONTINGENCY** — RDM alignment holds but attributes do not predict "
                 "held-out embeddings. Switch the primary endpoint to attribute-level "
                 "generalisation (BLUEPRINT_v2 4.1) and record it in the pre-registration.")
    else:
        L.append("\n**G4 FAIL** — re-run the census with the tactile-semantic encoder family "
                 "before training anything.")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tactus.eval.census")
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--vtd", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-perm-rdm", type=int, default=5000)
    ap.add_argument("--n-perm-lovo", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    vtd = load_vtd(args.vtd).sort_values("video_id").reset_index(drop=True)
    npzs = sorted(p for p in args.emb_dir.glob("*.npz"))
    if not npzs:
        print(f"no .npz under {args.emb_dir}", file=sys.stderr)
        return 2

    results = []
    for p in npzs:
        print(f"[census] {p.name} ...", flush=True)
        results.append(census_one(p, vtd, n_perm_rdm=args.n_perm_rdm,
                                  n_perm_lovo=args.n_perm_lovo, seed=args.seed))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "census.json").write_text(json.dumps(results, indent=2, default=str))
    report = render_report(results)
    (args.out / "census_report.md").write_text(report)
    print(report)
    print(f"[written] {args.out / 'census_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
