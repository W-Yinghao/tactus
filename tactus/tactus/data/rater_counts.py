#!/usr/bin/env python
"""Rebuild the VTD 90x4 rater-count table from the OSF validation data (D25).

The ds005662 release does not ship the per-rater categorical responses; they
live in the VTD project's OSF storage (https://osf.io/jvkqa/, Analysis/):
``DATA1.csv`` and ``DATA2.csv``.  This module replicates the published
``VTD_analysis.Rmd`` exactly:

- rows 1..175 of each file are kept (raters; DATA1 has 179, DATA2 176 -- the
  Rmd trims both to 175, so **each video is rated by 175 raters**, and the
  "350 raters" in ``losses/softclip.py``'s docstring is the total across the
  two disjoint batches, not the per-video count);
- columns 11..280 are kept (45 videos x 6 metrics; the 46th V-block in the raw
  files is dropped by the Rmd's slice);
- questionnaire order V1..V45 = DATA1, V46..V90 = DATA2;
- ``Video_overview.csv`` row *i* is questionnaire video *i* and carries
  ``Video_YouTube`` -- the ``video_id`` key every other table in this repo uses.

The rebuild is **verified against the published percentages** in
``table_all_videos.csv`` (rounded exactly as the Rmd rounds); any mismatch
raises, because a silently wrong mapping would poison the SoftCLIP targets.

CLI (writes counts CSV + SoftCLIP target/row-weight ``.npy``)::

    python -m tactus.data.rater_counts --dir $TACTUS_WORK/derived/vtd_validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

CATEGORIES = ("Neutral", "Pleasant", "Unpleasant", "Painful")
N_VIDEOS = 90
N_RATERS_PER_VIDEO = 175
_RMD_ROWS = slice(0, 175)      # R 1:175
_RMD_COLS = slice(10, 280)     # R 11:280, 0-based


def load_rater_counts(vtd_dir: Path) -> pd.DataFrame:
    """(90, 4) counts indexed by ``video_id`` (1..90), columns ``CATEGORIES``."""
    frames = []
    for fname in ("DATA1.csv", "DATA2.csv"):
        df = pd.read_csv(vtd_dir / fname).iloc[_RMD_ROWS, _RMD_COLS]
        cat = df[[c for c in df.columns if c.endswith("_category")]]
        if cat.shape != (N_RATERS_PER_VIDEO, 45):
            raise RuntimeError(f"{fname}: category block is {cat.shape}, expected (175, 45)")
        frames.append(cat)
    cat_all = pd.concat(frames, axis=1)          # questionnaire order V1..V90
    cat_all.columns = range(1, N_VIDEOS + 1)

    overview = pd.read_csv(vtd_dir / "Video_overview.csv").iloc[:N_VIDEOS]
    q_to_yt = overview["Video_YouTube"].astype(int).to_numpy()
    if sorted(q_to_yt.tolist()) != list(range(1, N_VIDEOS + 1)):
        raise RuntimeError("Video_overview does not map questionnaire order onto video_id 1..90")

    counts = pd.DataFrame(0, index=range(1, N_VIDEOS + 1), columns=list(CATEGORIES))
    counts.index.name = "video_id"
    for q in range(1, N_VIDEOS + 1):
        vc = cat_all[q].value_counts()
        unknown = set(vc.index) - set(CATEGORIES)
        if unknown:
            raise RuntimeError(f"questionnaire video {q}: unknown categories {unknown}")
        if int(vc.sum()) != N_RATERS_PER_VIDEO:
            raise RuntimeError(f"questionnaire video {q}: {int(vc.sum())} responses, expected 175")
        for c in CATEGORIES:
            counts.loc[int(q_to_yt[q - 1]), c] = int(vc.get(c, 0))
    return counts


def verify_against_published(counts: pd.DataFrame, vtd_dir: Path) -> int:
    """Compare with table_all_videos.csv percentages; return videos checked."""
    pub = pd.read_csv(vtd_dir / "table_all_videos.csv")
    pub = pub.set_index(pub["Video_YouTube"].astype(int))
    n_checked = 0
    for vid in counts.index:
        ours = counts.loc[vid].to_numpy(dtype=float)
        ours_pct = np.round(ours / ours.sum() * 100.0)
        theirs = pub.loc[vid, list(CATEGORIES)].to_numpy(dtype=float)
        if not np.array_equal(ours_pct, theirs):
            raise RuntimeError(
                f"video_id {vid}: rebuilt percentages {ours_pct.tolist()} != "
                f"published {theirs.tolist()} -- the questionnaire->video_id "
                f"mapping or the Rmd slice is wrong; refusing to write targets."
            )
        n_checked += 1
    return n_checked


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, required=True,
                    help="directory holding DATA1.csv/DATA2.csv/Video_overview.csv/table_all_videos.csv")
    args = ap.parse_args(argv)

    counts = load_rater_counts(args.dir)
    n = verify_against_published(counts, args.dir)

    from ..losses.softclip import SoftCLIP

    target = SoftCLIP.from_rater_counts(counts.to_numpy()).numpy()
    weights = SoftCLIP.disagreement_weights(counts.to_numpy()).numpy()

    counts.to_csv(args.dir / "rater_counts.csv")
    np.save(args.dir / "softclip_target_video90.npy", target)
    np.save(args.dir / "softclip_row_weights_video90.npy", weights)
    (args.dir / "rater_counts_manifest.json").write_text(json.dumps({
        "n_videos": int(len(counts)), "n_raters_per_video": N_RATERS_PER_VIDEO,
        "verified_against_published_pct": n,
        "target_metric": "bhattacharyya",
        "source": "osf.io/jvkqa Analysis/DATA{1,2}.csv, Rmd slice rows 1:175 cols 11:280",
    }, indent=2))
    print(f"rebuilt 90x4 counts, verified {n}/90 videos against published percentages")
    print(f"wrote rater_counts.csv, softclip_target_video90.npy, softclip_row_weights_video90.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
