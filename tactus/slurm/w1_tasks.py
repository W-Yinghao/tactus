#!/usr/bin/env python
"""Deterministic task table for the W1 fleet (prereg/W1_EXPLORATION_FROZEN.md).

``python slurm/w1_tasks.py N`` prints the shell command for task N; ``--table``
prints the whole mapping (also written next to the pool state as the audit
trail).  The five probe units (seed 0, fold 0, one per mechanism) are excluded
here because they already ran under their fleet names -- re-running would trip
run.py's mixing guard.

Fleet: 147 units.  Run-directory naming (run.name + '_' + tag):
  w1_codebook_ce_s{S}                     T-vid, live codebook          (P1)
  nice_protonce_w1_vid_s{S}  (S=1,2)      T-vid, frozen projector       (P1; seed 0 = nice_protonce__subj80)
  nice_protonce_w1_cap{full,vis,blind}_s{S}
  w1_supcon_material_s{S} / w1_rnc_affect_s{S}
  nice_protonce_w1_capvis_w{W}_s{S} / w1_rnc_affect_w{W}_s{S}   windows W=0,1,2
"""
from __future__ import annotations

import sys
from typing import List, Tuple

FOLDS = (0, 1, 2, 3)
SEEDS = (0, 1, 2)
WINDOWS = {0: "[0,40]", 1: "[40,80]", 2: "[80,120]"}
PY = "python -u -m tactus.train.run"
PROBED = {("codebook", 0, 0), ("rnc", 0, 0), ("supcon", 0, 0), ("capvis", 0, 0), ("capvis_w0", 0, 0)}


def _cmd(arm: str, fold: int, seed: int) -> str:
    ws = "--regime within_subject"
    if arm == "codebook":
        return f"{PY} -c configs/w1_codebook_ce.yaml {ws} --folds {fold} --tag s{seed} -o run.seed={seed}"
    if arm == "vid":
        return f"{PY} -c configs/nice_protonce.yaml {ws} --folds {fold} --tag w1_vid_s{seed} -o run.seed={seed}"
    if arm in ("capfull", "capvis", "capblind"):
        tgt = {"capfull": "full", "capvis": "vis", "capblind": "blind"}[arm]
        return (f"{PY} -c configs/nice_protonce.yaml {ws} --folds {fold} --tag w1_{arm}_s{seed} "
                f"-o video.model_tag=target-cap-{tgt} run.seed={seed}")
    if arm == "supcon":
        return f"{PY} -c configs/w1_supcon_material.yaml {ws} --folds {fold} --tag s{seed} -o run.seed={seed}"
    if arm == "rnc":
        return f"{PY} -c configs/w1_rnc_affect.yaml {ws} --folds {fold} --tag s{seed} -o run.seed={seed}"
    if arm.startswith("capvis_w"):
        w = int(arm[-1])
        return (f"{PY} -c configs/nice_protonce.yaml {ws} --folds {fold} --tag w1_capvis_w{w}_s{seed} "
                f"-o video.model_tag=target-cap-vis run.seed={seed} \"data.crop_samples={WINDOWS[w]}\"")
    if arm.startswith("rnc_w"):
        w = int(arm[-1])
        return (f"{PY} -c configs/w1_rnc_affect.yaml {ws} --folds {fold} --tag w{w}_s{seed} "
                f"-o run.seed={seed} \"data.crop_samples={WINDOWS[w]}\"")
    raise KeyError(arm)


def table() -> List[Tuple[int, str, int, int, str]]:
    arms = ["codebook", "vid", "capfull", "capvis", "capblind", "supcon", "rnc",
            "capvis_w0", "capvis_w1", "capvis_w2", "rnc_w0", "rnc_w1", "rnc_w2"]
    rows = []
    for arm in arms:
        seeds = (1, 2) if arm == "vid" else SEEDS      # seed 0 of T-vid ProtoNCE exists already
        for seed in seeds:
            for fold in FOLDS:
                if (arm, fold, seed) in PROBED:
                    continue
                rows.append((len(rows), arm, fold, seed, _cmd(arm, fold, seed)))
    return rows


def main() -> int:
    rows = table()
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        print("task,arm,fold,seed,cmd")
        for r in rows:
            print(",".join([str(r[0]), r[1], str(r[2]), str(r[3]), r[4].replace(",", ";")]))
        return 0
    n = int(sys.argv[1])
    print(rows[n][4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
