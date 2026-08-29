#!/usr/bin/env python
"""D24 capability table: text→EEG retrieval and zero-shot attribute prompts.

Implements prereg/D24_CAPABILITY_FROZEN.md and nothing beyond it.  Offline:
captions pass through each fold's own frozen video projector
(``checkpoints/best.pt: projector``, 768→1024→256) into the space the EEG was
trained to inhabit; no model is trained or updated here.

Stages::

    python -m tactus.eval.text_capability probe    # gates on vf00, 2 subjects
    python -m tactus.eval.text_capability run      # full 4-fold table

The tower gate (stimulus side, no EEG) decides whether a Table-A null is a
brain result or a blocked tower; it is computed per fold and the run REFUSES
to print Table A when the gate failed, per the frozen grid (cell A-3).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.events import load_vtd

RUN = "nice_protonce__subj80"
FOLDS = ("vf00", "vf01", "vf02", "vf03")   # vf04 sealed (D16)
N_PERM = 5000
SEED = 0
TOWER_GATE_TOP1 = 0.15

MATERIALS = ("metal", "wood", "plastic", "cotton", "fabric", "hair", "skin", "sponge")
PROMPTS = {
    "material": [f"a video of a hand being touched by something made of {m}" for m in MATERIALS],
    "approaching": ["a video of a hand with something approaching it",
                    "a video of a hand with something moving away from it"],
    "toucher": ["a video of a hand touched by another person's hand",
                "a video of a hand touched by a handheld object"],
    "valence": ["a video of a pleasant touch", "a video of a neutral touch",
                "a video of an unpleasant touch"],
}
BUNDLE_QUALIFIED = ("material", "toucher")  # D17


def _work() -> Path:
    return Path(os.environ.get("TACTUS_WORK", "/projects/EEG-foundation-model/tactus_work"))


def _runs() -> Path:
    return Path(os.environ.get("TACTUS_RESULTS_DIR", _work() / "results")) / "runs"


def _out_dir(out: Optional[Path]) -> Path:
    return out if out is not None else _work() / "results" / "text_capability"


# --------------------------------------------------------------------------- #
# projector + embeddings
# --------------------------------------------------------------------------- #
_PROJ_CACHE: Dict[str, dict] = {}


def _projector_sd(fold_dir: Path) -> dict:
    """The fold's video-projector weights, loaded once and structurally checked.

    heads.VideoProjector with n_layers=2 is ``Sequential(Linear, GELU, Dropout,
    Linear)`` -- state-dict keys net.0.* and net.3.*.  Anything else means the
    arm was built differently and this evaluator must not guess.
    """
    key = str(fold_dir)
    if key not in _PROJ_CACHE:
        import torch

        sd = torch.load(fold_dir / "checkpoints" / "best.pt", map_location="cpu",
                        weights_only=False)["projector"]
        if set(sd) != {"net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"}:
            raise RuntimeError(
                f"{fold_dir}: projector keys {sorted(sd)} do not match the "
                "Linear-GELU-Dropout-Linear contract; refusing to guess the "
                "architecture.")
        _PROJ_CACHE[key] = {k: v.numpy().astype(np.float64) for k, v in sd.items()}
    return _PROJ_CACHE[key]


def project_texts(fold_dir: Path, emb768: np.ndarray) -> np.ndarray:
    from scipy import special

    sd = _projector_sd(fold_dir)
    h = emb768 @ sd["net.0.weight"].T + sd["net.0.bias"]
    h = 0.5 * h * (1.0 + special.erf(h / np.sqrt(2.0)))      # exact GELU
    out = h @ sd["net.3.weight"].T + sd["net.3.bias"]
    return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def _centre(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return _l2(x - mean)


# --------------------------------------------------------------------------- #
# tower gate (stimulus side, no EEG)
# --------------------------------------------------------------------------- #
def tower_gate(fold_dir: Path, text90: np.ndarray, video90: np.ndarray,
               n_perm: int = N_PERM, seed: int = SEED) -> Dict[str, object]:
    t = project_texts(fold_dir, text90)
    v = project_texts(fold_dir, video90)
    out = {}
    for variant in ("centred", "raw"):
        tq = _centre(t, t.mean(axis=0)) if variant == "centred" else t
        vg = _centre(v, v.mean(axis=0)) if variant == "centred" else v
        sim = tq @ vg.T
        top1 = float((sim.argmax(axis=1) == np.arange(90)).mean())
        rng = np.random.default_rng(seed)
        null = np.array([(sim[:, rng.permutation(90)].argmax(axis=1)
                          == np.arange(90)).mean() for _ in range(n_perm)])
        p = float((1 + np.sum(null >= top1)) / (1 + n_perm))
        out[variant] = {"top1": top1, "p": p}
    prim = out["centred"]
    out["passed"] = bool(prim["top1"] >= TOWER_GATE_TOP1 and prim["p"] < 0.05)
    return out


# --------------------------------------------------------------------------- #
# per-fold evaluation
# --------------------------------------------------------------------------- #
def _fold_data(fold_dir: Path) -> Dict[str, np.ndarray]:
    z = np.load(fold_dir / "test_embeddings.npz")
    return {k: z[k] for k in z.files}


def eval_fold(fold_dir: Path, text90: np.ndarray, variant: str,
              subjects: Optional[Sequence[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tables A and B rows for one fold. ``text90`` is (90, 768), video_id order."""
    d = _fold_data(fold_dir)
    gal_vids = d["gallery_video_ids"].astype(int)          # 18 test videos
    t_all = project_texts(fold_dir, text90)                # (90, 256) L2
    t_mean = t_all.mean(axis=0)

    vtd = load_vtd(Path(os.environ.get(
        "TACTUS_BIDS_ROOT", "/projects/EEG-foundation-model/ds005662"))
        / "code" / "analysis" / "VTD.csv")
    from ..data.captions import _tercile_words
    val_terc = np.array(_tercile_words(vtd["valence"].to_numpy(float),
                                       ("unpleasant", "neutral", "pleasant")))
    # attribute prompt embeddings, per fold projector
    prompt_emb = {a: project_texts(fold_dir, _embed_prompts(PROMPTS[a]))
                  for a in PROMPTS}

    labels = {
        "material": vtd.set_index("video_id")["material"].astype(str),
        "approaching": vtd.set_index("video_id")["approaching"].astype(str),
        "toucher": vtd.set_index("video_id")["toucher"].astype(str),
        "valence": pd.Series(val_terc, index=vtd["video_id"].to_numpy()),
    }
    prompt_class = {
        "material": np.array(MATERIALS),
        "approaching": np.array(["yes", "no"]),
        "toucher": np.array(["hand", "object"]),
        "valence": np.array(["pleasant", "neutral", "unpleasant"]),
    }

    subj_ids = np.unique(d["subject_id"]) if subjects is None else np.asarray(subjects)
    rows_a, rows_b = [], []
    for sid in subj_ids:
        m = d["subject_id"] == sid
        z = d["z_eeg"][m].astype(np.float64)
        vids = d["video_id"][m].astype(int)
        missing = [int(v) for v in gal_vids if not (vids == v).any()]
        if missing:
            raise RuntimeError(f"subject {int(sid)}: no test trials for videos "
                               f"{missing} -- coverage gate, refusing NaN prototypes")
        protos = np.stack([_l2(z[vids == v].mean(axis=0)) for v in gal_vids])
        if variant == "centred":
            e_mean = _l2(z).mean(axis=0)
            protos_c = _centre(protos, e_mean)
            t_c = _centre(t_all, t_mean)
        else:
            protos_c, t_c = protos, _l2(t_all)
        tq = t_c[gal_vids - 1]                              # (18, 256)

        sim = tq @ protos_c.T                               # text -> EEG
        rows_a.append({"subject_id": int(sid), "direction": "text->eeg",
                       "grain": "prototype", "variant": variant,
                       "top1": float((sim.argmax(axis=1) == np.arange(len(gal_vids))).mean())})
        sim_m = protos_c @ tq.T                             # EEG -> text
        rows_a.append({"subject_id": int(sid), "direction": "eeg->text",
                       "grain": "prototype", "variant": variant,
                       "top1": float((sim_m.argmax(axis=1) == np.arange(len(gal_vids))).mean())})
        # k=1 companion: nearest single trial decides the video
        trials_c = _centre(_l2(z), e_mean) if variant == "centred" else _l2(z)
        near = trials_c @ tq.T                              # (n_trials, 18)
        pred_vid = gal_vids[near.argmax(axis=1)]            # not used: text->trial dir below
        sim_k1 = tq @ trials_c.T                            # (18, n_trials)
        top_trial = sim_k1.argmax(axis=1)
        rows_a.append({"subject_id": int(sid), "direction": "text->eeg",
                       "grain": "k1", "variant": variant,
                       "top1": float((vids[top_trial] == gal_vids).mean())})

        # Table B: attribute prompts on prototypes
        for attr, pe in prompt_emb.items():
            pe_c = _centre(pe, t_mean) if variant == "centred" else _l2(pe)
            pred = prompt_class[attr][(protos_c @ pe_c.T).argmax(axis=1)]
            truth = labels[attr].loc[gal_vids].to_numpy()
            classes = np.unique(truth)
            accs = [float((pred[truth == c] == c).mean()) for c in classes]
            rows_b.append({"subject_id": int(sid), "attribute": attr, "variant": variant,
                           "balanced_acc": float(np.mean(accs)),
                           "n_classes_present": int(len(classes)),
                           "majority_rate": float(pd.Series(truth).value_counts().iloc[0] / len(truth)),
                           "chance": 1.0 / len(prompt_class[attr])})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


_PROMPT_CACHE: Dict[Tuple[str, ...], np.ndarray] = {}


def _embed_prompts(prompts: List[str]) -> np.ndarray:
    key = tuple(prompts)
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    import torch
    from transformers import AutoModel, AutoProcessor

    from .tactile_spaces import SIGLIP_ID

    torch.manual_seed(0)
    model = AutoModel.from_pretrained(SIGLIP_ID)
    model.eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    with torch.no_grad():
        tok = proc(text=prompts, padding="max_length", truncation=True, return_tensors="pt")
        raw = model.get_text_features(**tok)
        if not torch.is_tensor(raw):
            for attr in ("text_embeds", "pooler_output"):
                if getattr(raw, attr, None) is not None:
                    raw = getattr(raw, attr)
                    break
    emb = raw.double().numpy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    _PROMPT_CACHE[key] = emb
    return emb


# --------------------------------------------------------------------------- #
# permutation inference for Table A (subject-aggregated, video-level null)
# --------------------------------------------------------------------------- #
def perm_p_table_a(fold_dirs: List[Path], text90: np.ndarray, variant: str,
                   direction: str, n_perm: int = N_PERM, seed: int = SEED) -> Dict[str, float]:
    """Permute the caption<->video assignment within fold; statistic = subject-mean top1."""
    rng = np.random.default_rng(seed)
    per_fold = []
    for fd in fold_dirs:
        d = _fold_data(fd)
        gal_vids = d["gallery_video_ids"].astype(int)
        t_all = project_texts(fd, text90)
        t_mean = t_all.mean(axis=0)
        sims = []
        for sid in np.unique(d["subject_id"]):
            m = d["subject_id"] == sid
            z = d["z_eeg"][m].astype(np.float64)
            vids = d["video_id"][m].astype(int)
            if any(not (vids == v).any() for v in gal_vids):
                raise RuntimeError(f"subject {int(sid)}: missing test videos")
            protos = np.stack([_l2(z[vids == v].mean(axis=0)) for v in gal_vids])
            if variant == "centred":
                protos = _centre(protos, _l2(z).mean(axis=0))
                tq = _centre(t_all, t_mean)[gal_vids - 1]
            else:
                tq = _l2(t_all)[gal_vids - 1]
            sims.append(tq @ protos.T if direction == "text->eeg" else protos @ tq.T)
        per_fold.append(np.stack(sims))                     # (n_subj, 18, 18)
    obs = float(np.mean([ (s.argmax(axis=2) == np.arange(s.shape[1])).mean()
                          for s in per_fold]))
    null = np.zeros(n_perm)
    for k in range(n_perm):
        acc = []
        for s in per_fold:
            perm = rng.permutation(s.shape[1])
            acc.append(float((s.argmax(axis=2) == perm[None, :]).mean()))
        null[k] = np.mean(acc)
    p = float((1 + np.sum(null >= obs)) / (1 + n_perm))
    return {"obs_top1": obs, "p": p, "n_perm": n_perm}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def _inputs():
    text90 = np.load(_work() / "derived" / "text_emb" / "siglip2-base-captions.npz")["text_emb"]
    video90 = np.load(_work() / "derived" / "video_emb" / "siglip2-base.npz")["base_emb"]
    video90 = video90 / np.linalg.norm(video90, axis=1, keepdims=True)
    return text90.astype(np.float64), video90.astype(np.float64)


def run_probe(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    text90, video90 = _inputs()
    fd = _runs() / RUN / "folds" / "vf00"
    failures: List[str] = []

    gate = tower_gate(fd, text90, video90, n_perm=1000)
    a1, b1 = eval_fold(fd, text90, "centred", subjects=None)
    a1 = a1[a1.subject_id.isin(sorted(a1.subject_id.unique())[:2])]
    a2, _ = eval_fold(fd, text90, "centred", subjects=sorted(
        set(_fold_data(fd)["subject_id"].astype(int).tolist()))[:2])
    x = a1[a1.grain == "prototype"].sort_values(["subject_id", "direction"]).top1.to_numpy()
    y = a2[a2.grain == "prototype"].sort_values(["subject_id", "direction"]).top1.to_numpy()
    if not np.array_equal(x, y):
        failures.append("gate2: evaluation not deterministic across rebuilds")
    if b1.n_classes_present.min() < 2:
        failures.append("gate3: an attribute has <2 classes among the 18 test videos")

    report = {"verdict": "PASS" if not failures else "FAIL",
              "failures": failures, "tower_gate_vf00": gate}
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def run_full(out_dir: Path) -> int:
    from scipy import stats

    out_dir.mkdir(parents=True, exist_ok=True)
    text90, video90 = _inputs()
    fold_dirs = [_runs() / RUN / "folds" / f for f in FOLDS]

    gates = {f: tower_gate(fd, text90, video90) for f, fd in zip(FOLDS, fold_dirs)}
    gate_pass = all(g["passed"] for g in gates.values())

    rows_a, rows_b = [], []
    for f, fd in zip(FOLDS, fold_dirs):
        for variant in ("centred", "raw"):
            a, b = eval_fold(fd, text90, variant)
            a["fold"], b["fold"] = f, f
            rows_a.append(a)
            rows_b.append(b)
    A = pd.concat(rows_a, ignore_index=True)
    B = pd.concat(rows_b, ignore_index=True)
    A.to_csv(out_dir / "table_a_raw_rows.csv", index=False)
    B.to_csv(out_dir / "table_b_raw_rows.csv", index=False)

    summary: Dict[str, object] = {"tower_gates": gates, "tower_gate_all_pass": gate_pass}
    if gate_pass:
        for direction in ("text->eeg", "eeg->text"):
            perm = perm_p_table_a(fold_dirs, text90, "centred", direction)
            sub = (A[(A.direction == direction) & (A.grain == "prototype")
                     & (A.variant == "centred")]
                   .groupby("subject_id").top1.mean())
            lo, hi = np.percentile(
                [np.random.default_rng(i).choice(sub, len(sub)).mean() for i in range(2000)],
                [2.5, 97.5])
            summary[direction] = {"subject_mean_top1": float(sub.mean()),
                                  "ci95": [float(lo), float(hi)],
                                  "chance": 1 / 18, **perm}
        k1 = (A[(A.direction == "text->eeg") & (A.grain == "k1") & (A.variant == "centred")]
              .groupby("subject_id").top1.mean())
        summary["text->eeg_k1"] = {"subject_mean_top1": float(k1.mean()), "chance": 1 / 18}
        battr = {}
        for attr, g in B[B.variant == "centred"].groupby("attribute"):
            per_sub = g.groupby("subject_id").balanced_acc.mean()
            battr[attr] = {"balanced_acc": float(per_sub.mean()),
                           "majority_rate": float(g.majority_rate.mean()),
                           "chance": float(g.chance.iloc[0]),
                           "bundle_qualified": attr in BUNDLE_QUALIFIED}
        summary["table_b"] = battr
    else:
        summary["verdict"] = "A-3: blocked at the tower; no brain-level conclusion"

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["probe", "run"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out_dir = _out_dir(args.out)
    return run_probe(out_dir) if args.stage == "probe" else run_full(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
