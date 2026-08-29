#!/usr/bin/env python
"""Deterministic captions from VTD attributes (D24, blueprint v3 structured-text rule).

One caption per base video, generated from the structured VTD columns by a
fixed template -- **never** from a free-running VLM (the blueprint rejects
those as uncontrolled and unattributable).  The same video under any of the 4
orientations keeps the same caption: flips change pixels, not the touch event.

Template (every clause traceable to one VTD column)::

    a third-person video of a hand: {description}, {touch_type}{ed} by
    {a toucher} with {an object} ({material} contact), {approaching|withdrawing};
    {valence word}, {threat word}[, painful]

``description`` is the fixed "Description of touch" string shipped in VTD.csv
(deterministic stimulus metadata, not VLM output -- the blueprint's own example
caption, "slowly stroked with a sponge", is this column).  Without it the
attribute grid alone leaves 26 of 90 captions duplicated (the D17 collinearity
fact in another costume), which the duplicate guard below documents.

Discretization is frozen here: valence terciles over the 90 videos map to
unpleasant/neutral/pleasant, threat terciles to low/moderate/high threat, and
``painful`` is appended when >=20% of raters called the video Painful (the
``pain`` column is the rater percentage).  Terciles are data-derived but from
public stimulus metadata only -- no EEG is involved anywhere in this module.

CLI (writes captions.csv and the SigLIP2 ``text_emb`` cache)::

    python -m tactus.data.captions --out $TACTUS_WORK/derived/text_emb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .events import load_vtd

SIGLIP_ID = "google/siglip2-base-patch16-224"

#: irregular past participles for the touch_type verbs; -ed otherwise.
_PARTICIPLE = {
    "grab": "grabbed", "pinch": "pinched", "pull": "pulled", "punch": "punched",
    "push": "pushed", "scratch": "scratched", "slap": "slapped",
    "slide": "slid across", "stab": "stabbed", "stroke": "stroked",
    "touch": "touched", "injection": "injected",
}
_PAIN_PCT_THRESHOLD = 20.0


def _article(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def _noun(raw: str) -> str:
    return raw.replace("_", " ")


def _tercile_words(values: np.ndarray, words: tuple) -> List[str]:
    lo, hi = np.quantile(values, [1 / 3, 2 / 3])
    return [words[0] if v <= lo else words[2] if v > hi else words[1] for v in values]


def _descriptions(vtd_path: Path) -> pd.Series:
    """The fixed 'Description of touch' column, keyed by video_id (load_vtd drops it)."""
    raw = pd.read_csv(vtd_path)
    id_col = next(c for c in raw.columns if c.strip().lower().startswith("youtube"))
    desc_col = next(c for c in raw.columns if "description" in c.strip().lower())
    return raw.set_index(raw[id_col].astype(int))[desc_col].astype(str).str.strip().str.lower()


def build_captions(vtd: pd.DataFrame, descriptions: Optional[pd.Series] = None) -> pd.DataFrame:
    """One deterministic caption per base video, keyed by ``video_id``."""
    val_w = _tercile_words(vtd["valence"].to_numpy(float),
                           ("unpleasant", "neutral", "pleasant"))
    thr_w = _tercile_words(vtd["threat"].to_numpy(float),
                           ("low threat", "moderate threat", "high threat"))
    rows = []
    for i, r in vtd.iterrows():
        verb = _PARTICIPLE[str(r["touch_type"])]
        obj = _noun(str(r["object"]))
        toucher = ("another person's hand" if r["toucher"] == "hand"
                   else f"{_article(obj)} {obj}")
        with_clause = "" if r["toucher"] == "hand" and r["object"] == "hand" \
            else f" with {_article(obj)} {obj}"
        if r["toucher"] == "object":
            with_clause = ""
        motion = "approaching" if r["approaching"] == "yes" else "withdrawing"
        desc = ""
        if descriptions is not None:
            desc = f"{descriptions.loc[int(r['video_id'])]}, "
        caption = (
            f"a third-person video of a hand: {desc}being {verb} by {toucher}"
            f"{with_clause} ({_noun(str(r['material']))} contact), {motion}; "
            f"{val_w[i]}, {thr_w[i]}"
        )
        if float(r["pain"]) >= _PAIN_PCT_THRESHOLD:
            caption += ", painful"
        rows.append({"video_id": int(r["video_id"]), "caption": caption})
    out = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)
    out["caption_class"] = out.groupby("caption").ngroup()
    if out["caption"].duplicated().any():
        dup = out[out.caption.duplicated(keep=False)]
        raise RuntimeError(
            f"{sorted(dup.video_id.tolist())} share captions -- the template "
            "does not separate the stimuli; pass the descriptions series "
            "(default CLI behaviour) or accept class-level captions explicitly."
        )
    return out


def embed_captions(captions: pd.DataFrame) -> np.ndarray:
    """(90, D) SigLIP2 text-tower embeddings, L2 rows, video_id order."""
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.manual_seed(0)
    model = AutoModel.from_pretrained(SIGLIP_ID)
    model.eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    outs = []
    with torch.no_grad():
        for start in range(0, len(captions), 16):
            chunk = captions["caption"].iloc[start:start + 16].tolist()
            tok = proc(text=chunk, padding="max_length", truncation=True,
                       return_tensors="pt")
            raw = model.get_text_features(**tok)
            if not torch.is_tensor(raw):
                for attr in ("text_embeds", "pooler_output"):
                    if getattr(raw, attr, None) is not None:
                        raw = getattr(raw, attr)
                        break
            outs.append(raw.double().numpy())
    emb = np.concatenate(outs, axis=0)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vtd", type=Path,
                    default=Path("/projects/EEG-foundation-model/ds005662/code/analysis/VTD.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    vtd = load_vtd(args.vtd)
    caps = build_captions(vtd, descriptions=_descriptions(args.vtd))
    emb = embed_captions(caps)

    caps.to_csv(args.out / "captions.csv", index=False)
    # text_emb is per base video; conditions replicate it 4x by contract
    # (condition_id = (video_id-1)*4 + orientation; flips keep the caption).
    cond = np.repeat(emb, 4, axis=0)
    np.savez(args.out / "siglip2-base-captions.npz",
             text_emb=emb, cond_text_emb=cond,
             video_id=caps["video_id"].to_numpy())
    (args.out / "captions_manifest.json").write_text(json.dumps({
        "template": "structured VTD attributes, frozen in tactus/data/captions.py",
        "n_captions": int(len(caps)),
        "pain_pct_threshold": _PAIN_PCT_THRESHOLD,
        "example": caps["caption"].iloc[0],
    }, indent=2))
    print(f"{len(caps)} captions; example: {caps['caption'].iloc[0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
