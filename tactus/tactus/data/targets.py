#!/usr/bin/env python
"""Alignment-target caches for the W1 target-space factor (EXPLORATION_PROGRAM_v2 §3 E2).

Each target is written in contract C's shape so ``load_video_emb`` accepts it
unchanged: ``cond_emb (360, D)``, ``base_emb (90, D)``, ``video_id``,
``orientation``, ``meta``.  The four orientations share a target -- flips
change pixels, not the described event.  Tags:

- ``target-cap-full``  D24 captions through the frozen SigLIP2 text tower;
- ``target-cap-vis``   visual-only captions (no material clause, no affect words);
- ``target-cap-blind`` material-blind captions (see captions.build_captions);
- ``target-mat``       one prompt per material class -> 8 unique vectors.

The text tower is frozen; nothing here is trained.  ``python -m tactus.data.targets``
writes all four into ``$TACTUS_WORK/derived/video_emb/``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .captions import _descriptions, build_captions, embed_captions
from .events import load_vtd

MATERIAL_PROMPT = "a video of a hand being touched by something made of {m}"
N_VIDEOS, N_ORI = 90, 4


def _write(out_dir: Path, tag: str, base: np.ndarray, note: dict) -> Path:
    assert base.shape[0] == N_VIDEOS, base.shape
    base = base / np.linalg.norm(base, axis=1, keepdims=True).clip(1e-12)
    cond = np.repeat(base, N_ORI, axis=0)                   # (video_id-1)*4 + orientation
    video_id = np.repeat(np.arange(1, N_VIDEOS + 1), N_ORI)
    orientation = np.tile(np.arange(N_ORI), N_VIDEOS)
    meta = {"model_tag": tag, "family": "text_target", "tower": "google/siglip2-base-patch16-224 (frozen text)",
            "built_utc": datetime.now(timezone.utc).isoformat(), **note}
    p = out_dir / f"{tag}.npz"
    np.savez(p, cond_emb=cond.astype(np.float32), base_emb=base.astype(np.float32),
             video_id=video_id, orientation=orientation, meta=json.dumps(meta))
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2))
    return p


def build_all(out_dir: Path, vtd_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vtd = load_vtd(vtd_path)
    desc = _descriptions(vtd_path)
    for variant in ("full", "vis", "blind"):
        caps = build_captions(vtd, descriptions=desc, variant=variant)
        emb = embed_captions(caps)
        _write(out_dir, f"target-cap-{variant}", emb,
               {"variant": variant, "n_unique_captions": int(caps.caption.nunique()),
                "example": caps.caption.iloc[0]})
        print(f"target-cap-{variant}: {caps.caption.nunique()}/90 unique; e.g. {caps.caption.iloc[0]!r}")
    # material prompts: 8 unique vectors, one per class, replicated per video
    import pandas as pd
    mats = sorted(vtd["material"].astype(str).unique())
    prompts = pd.DataFrame({"caption": [MATERIAL_PROMPT.format(m=m) for m in mats]})
    pe = embed_captions(prompts)                              # (8, D)
    idx = {m: i for i, m in enumerate(mats)}
    base = np.stack([pe[idx[str(m)]] for m in vtd.sort_values("video_id")["material"]])
    _write(out_dir, "target-mat", base, {"materials": mats, "prompt": MATERIAL_PROMPT})
    print(f"target-mat: {len(mats)} classes -> 90 rows")


def main() -> int:
    work = Path(os.environ.get("TACTUS_WORK", "/projects/EEG-foundation-model/tactus_work"))
    bids = Path(os.environ.get("TACTUS_BIDS_ROOT", "/projects/EEG-foundation-model/ds005662"))
    build_all(work / "derived" / "video_emb", bids / "code" / "analysis" / "VTD.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
