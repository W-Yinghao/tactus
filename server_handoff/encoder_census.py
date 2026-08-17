#!/usr/bin/env python
"""TACTUS Phase 0 encoder census on the 90 original-orientation touch videos.

Three tests (BLUEPRINT_v2 §4.1):
  1) embedding-collapse check (mean pairwise cosine, PC spectrum)
  2) embedding RDM vs attribute RDMs (Spearman + video-level permutation p)
  3) attribute -> embedding leave-one-video-out predictability (ridge; retrieval
     rank of held-out video among 90) — decides video-level vs attribute-level
     zero-shot as the primary endpoint.

Usage: python encoder_census.py --stim stimuli --vtd ds005662/code/analysis/VTD.csv --out phase0_out
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import RidgeCV

N_PERM_RSA = 1000
N_PERM_LOVO = 100


def find_original_dir(stim_root: Path) -> Path:
    cands = [d for d in stim_root.iterdir() if d.is_dir()
             and "600ms" in d.name.lower()
             and "flip" not in d.name.lower()
             and "target" not in d.name.lower()]
    if len(cands) != 1:
        raise SystemExit(f"expected exactly one original-orientation folder under {stim_root}, got {cands}")
    return cands[0]


def read_frames(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {mp4} (check opencv/ffmpeg)")
    return frames


def embed_videos(files, model_name, device):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embs = []
    with torch.no_grad():
        for i, f in enumerate(files):
            frames = read_frames(f)
            inputs = proc(images=frames, return_tensors="pt").to(device)
            feat = model.get_image_features(**inputs)          # (n_frames, D)
            feat = torch.nn.functional.normalize(feat, dim=-1).mean(0)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            embs.append(feat.cpu().numpy())
            if (i + 1) % 15 == 0:
                print(f"  embedded {i + 1}/{len(files)}")
    return np.stack(embs)  # (90, D)


def rdm_from_emb(E):
    S = E @ E.T
    return 1.0 - S


def upper(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def perm_spearman(rdm_a, rdm_b, n_perm, rng):
    r_obs = stats.spearmanr(upper(rdm_a), upper(rdm_b))[0]
    null = np.empty(n_perm)
    n = rdm_a.shape[0]
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = stats.spearmanr(upper(rdm_a), upper(rdm_b[np.ix_(p, p)]))[0]
    pval = (np.sum(null >= r_obs) + 1) / (n_perm + 1)
    return r_obs, pval


def build_features(vtd):
    # object (28 classes on 90 videos) deliberately EXCLUDED: near-identifying -> inflates predictability
    cont = [c for c in ("valence", "arousal", "threat") if c in vtd.columns]
    cat = [c for c in ("material", "touch_type", "toucher", "approaching", "pain") if c in vtd.columns]
    X = []
    names = []
    for c in cont:
        v = pd.to_numeric(vtd[c], errors="coerce").to_numpy(dtype=float)
        X.append((v - np.nanmean(v)) / np.nanstd(v))
        names.append(c)
    for c in cat:
        d = pd.get_dummies(vtd[c].astype(str))
        for col in d.columns:
            X.append(d[col].to_numpy(dtype=float))
            names.append(f"{c}={col}")
    return np.column_stack(X), names, cont, cat


def lovo_retrieval(X, E, rng, n_perm):
    n = E.shape[0]

    def run(Xm):
        ranks = np.empty(n, dtype=int)
        for i in range(n):
            tr = np.arange(n) != i
            reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xm[tr], E[tr])
            pred = reg.predict(Xm[i:i + 1])[0]
            pred = pred / (np.linalg.norm(pred) + 1e-12)
            sims = E @ pred
            ranks[i] = int(np.sum(sims > sims[i])) + 1  # rank of true video, 1 = best
        return ranks

    ranks = run(X)
    null_top5 = np.empty(n_perm)
    for k in range(n_perm):
        null_top5[k] = float(np.mean(run(X[rng.permutation(n)]) <= 5))
    obs_top5 = float(np.mean(ranks <= 5))
    pval = (np.sum(null_top5 >= obs_top5) + 1) / (n_perm + 1)
    return ranks, obs_top5, null_top5, pval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", type=Path, required=True)
    ap.add_argument("--vtd", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("phase0_out"))
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", args.model)

    vdir = find_original_dir(args.stim)
    files = sorted(vdir.glob("*.mp4"), key=lambda f: int(re.match(r"^(\d+)\.", f.name).group(1)))
    vids = [int(re.match(r"^(\d+)\.", f.name).group(1)) for f in files]
    print(f"[census] {len(files)} original videos from {vdir}")
    if len(files) != 90:
        print(f"  WARNING: expected 90, got {len(files)} — proceeding, report will flag this")

    vtd = pd.read_csv(args.vtd)
    vtd.columns = [c.strip().lower().replace(" ", "_") for c in vtd.columns]
    id_col = next(c for c in vtd.columns if "youtube_video" in c or c in ("video", "id"))
    vtd = vtd.set_index(pd.to_numeric(vtd[id_col], errors="coerce")).loc[vids].reset_index(drop=True)

    E = embed_videos(files, args.model, args.device)
    np.save(args.out / f"embeddings_{tag}.npy", E)

    rep = [f"# Encoder census — {args.model}", f"videos={len(files)} dim={E.shape[1]}"]

    # 1) collapse check
    S = E @ E.T
    off = upper(S)
    _, sv, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    ev = sv ** 2 / np.sum(sv ** 2)
    collapse = off.mean() > 0.9 and ev[:3].sum() > 0.8
    rep.append("\n## 1) 嵌入坍缩检查")
    rep.append(f"- 成对余弦: mean={off.mean():.3f} min={off.min():.3f} max={off.max():.3f}")
    rep.append(f"- PC 解释方差: top3={ev[:3].sum():.3f} top10={ev[:10].sum():.3f}")
    rep.append(f"- 判定: {'**坍缩警报 — 触发备胎编码器（§4.1）**' if collapse else '未坍缩'}")

    # 2) RDM vs attribute RDMs
    rdm_e = rdm_from_emb(E)
    pd.DataFrame(rdm_e).to_csv(args.out / f"rdm_embed_{tag}.csv", index=False)
    rep.append("\n## 2) 嵌入 RDM vs 属性 RDM（Spearman，视频级置换 p）")
    any_sig = False
    for c in ("valence", "arousal", "threat"):
        if c in vtd.columns:
            v = pd.to_numeric(vtd[c], errors="coerce").to_numpy(dtype=float)
            z = (v - np.nanmean(v)) / np.nanstd(v)
            rdm_a = np.abs(z[:, None] - z[None, :])
            r, p = perm_spearman(rdm_e, rdm_a, N_PERM_RSA, rng)
            any_sig |= p < 0.05
            rep.append(f"- {c}: r={r:.3f} p={p:.4f}")
    for c in ("material", "touch_type", "toucher"):
        if c in vtd.columns:
            lab = vtd[c].astype(str).to_numpy()
            rdm_a = (lab[:, None] != lab[None, :]).astype(float)
            r, p = perm_spearman(rdm_e, rdm_a, N_PERM_RSA, rng)
            any_sig |= p < 0.05
            rep.append(f"- {c}: r={r:.3f} p={p:.4f}")
    rep.append(f"- 判定: {'≥1 属性显著 → Phase 0 go 条件满足' if any_sig else '**全不显著 — 换编码器族再试；仍不显著则蓝图 go/no-go 触发**'}")

    # 3) attribute -> embedding LOVO predictability
    X, names, cont, cat = build_features(vtd)
    rep.append("\n## 3) 属性→嵌入 LOVO 可预测性（决定主终点）")
    rep.append(f"- 特征: {len(names)} 维（连续 {cont} + one-hot {cat}；object 因近可识别性被排除）")
    ranks, top5, null_top5, pval = lovo_retrieval(X, E, rng, N_PERM_LOVO)
    top1 = float(np.mean(ranks <= 1))
    rep.append(f"- 检索: top-1={top1:.3f} top-5={top5:.3f} 平均秩={ranks.mean():.1f}/90")
    rep.append(f"- 置换零分布 top-5: mean={null_top5.mean():.3f}，p={pval:.4f}")
    if pval < 0.05:
        rep.append("- 判定: 属性可预测 held-out 嵌入 → **视频级零样本原理上可行**，按原案执行")
    else:
        rep.append("- 判定: **属性预测不了 held-out 嵌入 → 视频级零样本不可行，主终点改为属性级泛化（蓝图预案）**")

    (args.out / f"census_report_{tag}.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"\n[written] {args.out}/census_report_{tag}.md")


if __name__ == "__main__":
    main()
