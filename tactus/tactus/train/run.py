"""CLI entry point: run one config over every fold of a regime.

    python -m tactus.train.run --config configs/nice_infonce.yaml
    python -m tactus.train.run --config configs/nice_infonce.yaml \
        --override loss.name=protonce train.epochs=40 run.name=protonce_40

Multi-GPU is fold-level, not batch-level.  A single fold is 1-2 GPU-hours on a
0.1-4M parameter encoder, and the protocol asks for 40 of them, so the useful
parallelism is "one fold per GPU", which needs no distributed training code and
no gradient synchronisation:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python -m tactus.train.run \
          --config configs/nice_infonce.yaml --shard $i --num-shards 4 &
    done; wait

Each shard writes into its own per-fold directory, so the shards never collide,
and ``--aggregate-only`` afterwards collects whatever finished.  Every fold is
skipped if its ``done.json`` exists, which makes re-running the exact same
command line a safe way to resume an interrupted sweep.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from tactus.common import REPO_ROOT, atomic_write_json, hash_obj, load_trials, results_dir

log = logging.getLogger("tactus.train.run")

__all__ = ["main", "load_config", "run_sweep", "aggregate"]

DEFAULT_CONFIG = "configs/default.yaml"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> Any:
    """Load a YAML config, resolve its ``base:`` chain, and apply dot-list overrides.

    A config may name a parent with ``base: default.yaml`` (resolved relative to
    the config's own directory, then to the repo root).  Parents are merged
    parent-first, so a child only has to state what it changes -- which is the
    point of the whole schema: swapping the algorithm is one key.
    """
    from omegaconf import OmegaConf

    path = Path(path)
    if not path.is_absolute() and not path.exists():
        cand = REPO_ROOT / path
        if cand.exists():
            path = cand
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    chain: List[Path] = []
    seen: set = set()
    cur: Path | None = path
    while cur is not None:
        rp = cur.resolve()
        if rp in seen:
            raise ValueError(f"circular base: chain at {rp}")
        seen.add(rp)
        chain.append(rp)
        node = OmegaConf.load(rp)
        base = node.get("base", None) if hasattr(node, "get") else None
        if base is None:
            cur = None
        else:
            b = Path(str(base))
            cur = b if b.is_absolute() else (rp.parent / b)
            if not cur.exists():
                cur = REPO_ROOT / b

    cfg = OmegaConf.create({})
    for p in reversed(chain):  # parents first
        cfg = _merge_layer(cfg, OmegaConf.load(p))
    cfg.pop("base", None)
    if overrides:
        cfg = _merge_layer(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def _merge_layer(cfg: Any, layer: Any) -> Any:
    """Merge one config layer, with a special rule for the ``loss`` block.

    Losses have *disjoint* constructor kwargs, so deep-merging the loss block
    across layers is actively wrong: inheriting ``temperature: 0.04`` from the
    default config into a ``composite`` loss hands ``CompositeLoss`` a keyword it
    does not accept, and the run dies at construction.  Worse, when the stale key
    happens to be accepted, it silently changes the objective.

    The rule: **a layer that names a loss owns the entire loss block.**  So

        --override loss.name=protonce

    really is the one-key swap it looks like -- the inherited InfoNCE keywords
    are dropped rather than smuggled into ProtoNCE's constructor.  A layer that
    sets loss keys *without* naming a loss (``--override loss.temperature=0.02``)
    still merges normally, because that is a tweak, not a swap.
    """
    from omegaconf import OmegaConf

    def _name_of(node: Any) -> Any:
        block = node["loss"] if (node is not None and "loss" in node) else None
        if block is None:
            return None
        return block.get("name", None) if hasattr(block, "get") else None

    old_name, new_name = _name_of(cfg), _name_of(layer)
    merged = OmegaConf.merge(cfg, layer)
    if new_name is not None and old_name is not None and str(new_name) != str(old_name):
        merged["loss"] = OmegaConf.create(
            OmegaConf.to_container(layer["loss"], resolve=True)
        )
    return merged


def _container(cfg: Any) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #


def fold_key(fold: Any) -> str:
    """Stable directory name for a fold."""
    vf = getattr(fold, "video_fold_id", None)
    sf = getattr(fold, "subject_fold_id", None)
    if sf is None:
        return f"vf{int(vf):02d}"
    return f"vf{int(vf):02d}_sf{int(sf):02d}"


def build_folds(cfg: Any, trials: pd.DataFrame) -> List[Any]:
    """Call the project's ``make_folds`` (contract E)."""
    from tactus.data.splits import make_folds

    regime = str(cfg.eval.regime)
    folds = make_folds(
        trials,
        regime,
        n_video_folds=int(cfg.eval.get("n_video_folds", 5)),
        n_subject_folds=int(cfg.eval.get("n_subject_folds", 8)),
        seed=int(cfg.run.get("split_seed", cfg.run.get("seed", 0))),
    )
    log.info("regime=%s -> %d folds", regime, len(folds))
    return list(folds)


def select_folds(
    folds: Sequence[Any],
    *,
    only: Sequence[int] | None = None,
    shard: int = 0,
    num_shards: int = 1,
    max_folds: int | None = None,
) -> List[tuple]:
    """``[(index, fold), ...]`` after ``--folds`` / ``--shard`` / ``--max-folds``."""
    idx = list(range(len(folds)))
    if only:
        keep = set(int(i) for i in only)
        idx = [i for i in idx if i in keep]
    if num_shards > 1:
        idx = [i for i in idx if i % num_shards == shard]
    if max_folds is not None:
        idx = idx[: int(max_folds)]
    return [(i, folds[i]) for i in idx]


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple:
    """Percentile bootstrap CI over the *subject* axis.

    The resampling unit is the subject because model-ranking claims are
    fixed-stimulus inferences over an 80-subject sample.  Anything that
    generalizes over *stimuli* needs the video-level permutation in
    ``tactus.eval``, not this.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(int(n_boot), v.size), replace=True).mean(axis=1)
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def aggregate(run_root: Path | str, *, primary: str | None = None, seed: int = 0) -> Dict[str, Any]:
    """Collect every finished fold under ``run_root`` into summary tables.

    Writes ``summary_folds.csv`` (one row per fold), ``summary_subjects.csv``
    (one row per subject, averaged over the folds in which that subject was
    held out) and ``aggregate.json`` (the headline numbers with bootstrap CIs).
    """
    run_root = Path(run_root)
    fold_rows: List[Dict[str, Any]] = []
    per_subject: List[pd.DataFrame] = []
    for mfile in sorted(run_root.glob("folds/*/metrics.json")):
        with open(mfile, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        row = {k: v for k, v in blob.items() if not isinstance(v, (dict, list))}
        row.update({k: v for k, v in (blob.get("metrics") or {}).items()})
        row["fold"] = mfile.parent.name
        fold_rows.append(row)
        ps = mfile.parent / "per_subject.csv"
        if ps.exists():
            df = pd.read_csv(ps)
            df["fold"] = mfile.parent.name
            per_subject.append(df)

    out: Dict[str, Any] = {"n_folds": len(fold_rows), "run_root": str(run_root)}
    if not fold_rows:
        log.warning("no finished folds under %s", run_root)
        return out

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(run_root / "summary_folds.csv", index=False)

    metric_cols = [c for c in folds_df.columns if c.startswith("test/")]
    out["fold_means"] = {c: float(folds_df[c].mean(skipna=True)) for c in metric_cols}

    if per_subject:
        subj = pd.concat(per_subject, ignore_index=True)
        subj.to_csv(run_root / "summary_subjects.csv", index=False)
        stat_cols = [c for c in ("top1", "top5", "mean_rank") if c in subj.columns]
        grouped = (
            subj.groupby(["unit", "trial_type", "subject_id"], observed=True)[stat_cols]
            .mean()
            .reset_index()
        )
        subject_level: Dict[str, Any] = {}
        for (unit, ttype), g in grouped.groupby(["unit", "trial_type"], observed=True):
            for c in stat_cols:
                lo, hi = _bootstrap_ci(g[c].to_numpy(), seed=seed)
                subject_level[f"{unit}/{ttype}/{c}"] = {
                    "mean": float(g[c].mean()),
                    "sd": float(g[c].std(ddof=1)) if len(g) > 1 else float("nan"),
                    "n_subjects": int(len(g)),
                    "ci95": [lo, hi],
                }
        out["subject_level"] = subject_level

    if primary and primary in folds_df.columns:
        vals = folds_df[primary].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(vals, seed=seed)
        out["primary"] = {"metric": primary, "mean": float(np.nanmean(vals)),
                          "n_folds": int(np.isfinite(vals).sum()), "ci95_fold_level": [lo, hi]}
    atomic_write_json(run_root / "aggregate.json", out)
    log.info("aggregated %d folds -> %s", len(fold_rows), run_root / "aggregate.json")
    return out


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


def run_sweep(cfg: Any, args: argparse.Namespace) -> int:
    """Train every selected fold.  Returns the number of failed folds."""
    from tactus.common import EpochStore, load_video_emb
    from tactus.train.trainer import train_fold

    cfg_dict = _container(cfg)
    run_name = str(cfg.run.name) + (f"_{args.tag}" if args.tag else "")
    run_root = Path(args.out_root) if args.out_root else results_dir() / "runs"
    run_root = run_root / run_name
    (run_root / "folds").mkdir(parents=True, exist_ok=True)

    cfg_hash = hash_obj(cfg_dict)
    hash_file = run_root / "config_hash.txt"
    if hash_file.exists():
        old = hash_file.read_text(encoding="utf-8").strip()
        if old != cfg_hash and not args.force:
            log.warning(
                "config hash changed (%s -> %s) but folds from the old config are "
                "present in %s. Re-running will MIX results. Use --force to "
                "overwrite, or a fresh --tag / run.name.",
                old, cfg_hash, run_root,
            )
    hash_file.write_text(cfg_hash, encoding="utf-8")
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, run_root / "config.yaml")
    atomic_write_json(
        run_root / "env.json",
        {
            "argv": sys.argv, "host": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": time.time(), "config_hash": cfg_hash,
        },
    )

    subjects = cfg.data.get("subjects", None)
    trials = load_trials(
        subjects=None if subjects is None else [int(s) for s in subjects],
        exclude_post_target=bool(cfg.data.get("exclude_post_target", True)),
    )
    log.info("trial table: %d trials, %d subjects, %d videos",
             len(trials), trials["subject_id"].nunique(), trials["video_id"].nunique())

    folds = build_folds(cfg, trials)
    selected = select_folds(
        folds, only=args.folds, shard=args.shard, num_shards=args.num_shards,
        max_folds=args.max_folds,
    )
    log.info("this process will run %d/%d folds (shard %d/%d)",
             len(selected), len(folds), args.shard, args.num_shards)

    if args.dry_run:
        for i, f in selected:
            print(
                f"[{i:3d}] {fold_key(f)} regime={getattr(f, 'regime', '?')} "
                f"train_uid={len(getattr(f, 'train_uid', []))} "
                f"test_uid={len(getattr(f, 'test_uid', []))} "
                f"test_videos={len(getattr(f, 'test_videos', []))} "
                f"test_subjects={len(getattr(f, 'test_subjects', []))}"
            )
        return 0

    store = EpochStore(str(cfg.data.window))
    video = load_video_emb(str(cfg.video.model_tag))

    n_failed = 0
    for i, fold in selected:
        key = fold_key(fold)
        fold_dir = run_root / "folds" / key
        log.info("=== fold %d/%d: %s ===", i + 1, len(folds), key)
        try:
            res = train_fold(
                cfg_dict, fold, trials, fold_dir, device=args.device,
                resume=not args.no_resume, force=args.force, store=store, video=video,
            )
            log.info("fold %s: %s best=%.4f @epoch %d (%.1f min)",
                     key, res.status, res.best_monitor, res.best_epoch, res.seconds / 60)
        except KeyboardInterrupt:
            log.warning("interrupted -- last.pt is on disk, rerun the same command to resume")
            raise
        except Exception as exc:  # unattended: log, mark, keep going
            n_failed += 1
            log.error("fold %s FAILED: %s", key, exc)
            traceback.print_exc()
            atomic_write_json(
                fold_dir / "failed.json",
                {"error": repr(exc), "traceback": traceback.format_exc(), "time": time.time()},
            )
            if args.fail_fast:
                break

    if args.shard == 0 or args.num_shards == 1:
        aggregate(run_root, primary=str(cfg.eval.get("primary_metric", "")) or None,
                  seed=int(cfg.run.get("seed", 0)))
    return n_failed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tactus.train.run",
        description="Train a TACTUS config over every fold of a regime.",
    )
    p.add_argument("--config", "-c", required=True, help="path to a YAML config")
    p.add_argument("--override", "-o", nargs="*", default=[], metavar="key=value",
                   help="OmegaConf dot-list overrides, e.g. loss.name=protonce train.epochs=40")
    p.add_argument("--regime", default=None,
                   help="shorthand for --override eval.regime=...")
    p.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: auto)")
    p.add_argument("--folds", type=lambda s: [int(x) for x in s.split(",") if x != ""],
                   default=None, help="comma-separated fold indices to run")
    p.add_argument("--shard", type=int, default=0, help="this process's shard index")
    p.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--tag", default=None, help="suffix appended to run.name")
    p.add_argument("--out-root", default=None, help="override results/runs")
    p.add_argument("--force", action="store_true", help="re-run folds that already finished")
    p.add_argument("--no-resume", action="store_true", help="ignore checkpoints/last.pt")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failing fold")
    p.add_argument("--dry-run", action="store_true", help="print the fold plan and exit")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip training, just re-aggregate an existing run directory")
    p.add_argument("--print-config", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    overrides = list(args.override or [])
    if args.regime:
        overrides.append(f"eval.regime={args.regime}")
    cfg = load_config(args.config, overrides)

    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg))
        return 0

    if args.aggregate_only:
        run_name = str(cfg.run.name) + (f"_{args.tag}" if args.tag else "")
        root = Path(args.out_root) if args.out_root else results_dir() / "runs"
        aggregate(root / run_name,
                  primary=str(cfg.eval.get("primary_metric", "")) or None,
                  seed=int(cfg.run.get("seed", 0)))
        return 0

    n_failed = run_sweep(cfg, args)
    if n_failed:
        log.error("%d fold(s) failed -- see failed.json in the fold directories", n_failed)
    return 1 if n_failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
