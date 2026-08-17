#!/usr/bin/env python
"""TACTUS environment probe.

Stdlib-only: it must not crash on a machine missing every scientific package.
Reports what exists, what is missing, and the GPU inventory, then prints a
single install line for the gaps.

RUN IT ON A COMPUTE NODE — a slurm login node usually has no GPUs visible, so
probing there reports "no CUDA" even on a healthy cluster:

    srun -p L40S --gres=gpu:1 -t 00:10:00 --pty python env_probe.py --skip-disk

Storage lives at /projects/EEG-foundation-model (space already confirmed
sufficient), so --skip-disk is the normal mode here.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# package import name -> (pip name, why TACTUS needs it, hard requirement?)
REQUIREMENTS = {
    "torch":        ("torch", "EEG encoders, contrastive training", True),
    "numpy":        ("numpy", "everywhere", True),
    "pandas":       ("pandas", "trial table, events parsing", True),
    "pyarrow":      ("pyarrow", "trials.parquet", True),
    "scipy":        ("scipy", "stats, permutation", True),
    "sklearn":      ("scikit-learn", "linear MVPA baselines, ridge, LDA", True),
    "mne":          ("mne", "BDF reading, filtering, epoching, ICA", True),
    "cv2":          ("opencv-python-headless", "mp4 decoding for video embeddings", True),
    "transformers": ("transformers", "SigLIP2 / VideoMAE / X-CLIP frozen encoders", True),
    "yaml":         ("pyyaml", "configs", True),
    "joblib":       ("joblib", "per-subject parallel preprocessing", True),
    "matplotlib":   ("matplotlib", "figures", False),
    "autoreject":   ("autoreject", "artifact policy (sensitivity analysis)", False),
    "omegaconf":    ("omegaconf", "config overrides (argparse fallback exists)", False),
    "statsmodels":  ("statsmodels", "mixed models for stimulus-generalizing claims", False),
    "seaborn":      ("seaborn", "figures", False),
    "h5py":         ("h5py", "optional cache format", False),
    "picard":       ("python-picard", "faster ICA than fastica", False),
}

MIN_FREE_GB = 150.0  # 110 GB raw + 11 GB derivatives + derived epochs/embeddings headroom
DEFAULT_DATA_ROOT = "/projects/EEG-foundation-model"


def probe_packages() -> dict:
    out = {}
    for mod, (pip_name, why, hard) in REQUIREMENTS.items():
        rec = {"pip": pip_name, "why": why, "required": hard, "present": False, "version": None}
        try:
            importlib.import_module(mod)
            rec["present"] = True
            for cand in (pip_name, mod):
                try:
                    rec["version"] = md.version(cand)
                    break
                except md.PackageNotFoundError:
                    continue
        except Exception as e:  # ImportError, or a broken install raising something else
            rec["error"] = f"{type(e).__name__}: {e}"
        out[mod] = rec
    return out


def probe_torch() -> dict:
    info = {"available": False}
    try:
        import torch
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    info.update(
        available=True,
        version=torch.__version__,
        cuda_compiled=getattr(torch.version, "cuda", None),
        cuda_available=bool(torch.cuda.is_available()),
        device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        bf16_supported=bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if torch.cuda.is_available() else False,
    )
    gpus = []
    for i in range(info["device_count"]):
        try:
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({
                "index": i, "name": p.name,
                "total_gb": round(p.total_memory / 2**30, 1),
                "free_gb": round(free / 2**30, 1),
                "capability": f"{p.major}.{p.minor}",
                "multiprocessors": p.multi_processor_count,
            })
        except Exception as e:
            gpus.append({"index": i, "error": str(e)})
    info["gpus"] = gpus
    return info


def probe_slurm() -> dict:
    """Where are we running: login node or inside an allocation?"""
    inside = {k: v for k, v in os.environ.items()
              if k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                       "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS",
                       "SLURM_ARRAY_TASK_ID", "CUDA_VISIBLE_DEVICES")}
    return {"in_allocation": bool(inside.get("SLURM_JOB_ID")), "env": inside,
            "sbatch_available": shutil.which("sbatch") is not None}


def probe_system(data_root: Path, skip_disk: bool) -> dict:
    total_ram = None
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_ram = round(int(line.split()[1]) / 2**20, 1)
                    break
    except Exception:
        pass
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "cpu_count": os.cpu_count(),
        "ram_gb": total_ram,
        "disk_checked": not skip_disk,
    }
    if not skip_disk:
        probe_dir = data_root if data_root.exists() else Path.cwd()
        du = shutil.disk_usage(probe_dir)
        info.update(disk_probe_path=str(probe_dir),
                    disk_total_gb=round(du.total / 2**30, 1),
                    disk_free_gb=round(du.free / 2**30, 1),
                    disk_ok_for_full_download=du.free / 2**30 >= MIN_FREE_GB)
    return info


def probe_cli() -> dict:
    out = {}
    for tool, args in (("aws", ["aws", "--version"]),
                       ("datalad", ["datalad", "--version"]),
                       ("git-annex", ["git-annex", "version"]),
                       ("ffmpeg", ["ffmpeg", "-version"]),
                       ("nvidia-smi", ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                       "--format=csv,noheader"])):
        path = shutil.which(args[0])
        rec = {"path": path, "present": path is not None}
        if path:
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=20)
                rec["output"] = (r.stdout or r.stderr).strip().splitlines()[:4]
            except Exception as e:
                rec["output"] = [f"{type(e).__name__}: {e}"]
        out[tool] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    ap.add_argument("--skip-disk", action="store_true",
                    help="storage headroom already confirmed (normal mode on this cluster)")
    ap.add_argument("--out", type=Path, default=Path("env_report.json"))
    args = ap.parse_args()

    report = {
        "system": probe_system(args.data_root, args.skip_disk),
        "slurm": probe_slurm(),
        "packages": probe_packages(),
        "torch": probe_torch(),
        "cli": probe_cli(),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    sysinfo, pkgs, tor, cli = report["system"], report["packages"], report["torch"], report["cli"]
    slurm = report["slurm"]
    print("=" * 68)
    print("TACTUS ENVIRONMENT PROBE")
    print("=" * 68)
    print(f"python {sysinfo['python']}  ({sysinfo['python_exe']})")
    print(f"env: conda={sysinfo['conda_env']} venv={sysinfo['virtual_env']}")
    print(f"cpu={sysinfo['cpu_count']}  ram={sysinfo['ram_gb']} GB  platform={sysinfo['platform']}")
    if slurm["in_allocation"]:
        print(f"slurm: INSIDE allocation job={slurm['env'].get('SLURM_JOB_ID')} "
              f"partition={slurm['env'].get('SLURM_JOB_PARTITION')} "
              f"gpus={slurm['env'].get('SLURM_JOB_GPUS') or slurm['env'].get('CUDA_VISIBLE_DEVICES')}")
    elif slurm["sbatch_available"]:
        print("slurm: LOGIN NODE (sbatch present, no allocation)")
        print("  !! GPU results below are meaningless here. Re-run inside an allocation:")
        print("     srun -p L40S --gres=gpu:1 -t 00:10:00 --pty python env_probe.py --skip-disk")
    if sysinfo["disk_checked"]:
        print(f"disk @ {sysinfo['disk_probe_path']}: {sysinfo['disk_free_gb']} GB free / "
              f"{sysinfo['disk_total_gb']} GB total")
        if not sysinfo["disk_ok_for_full_download"]:
            print(f"  !! need >= {MIN_FREE_GB} GB for the full raw download.")

    print("\n-- GPU --")
    if tor.get("cuda_available"):
        for g in tor["gpus"]:
            print(f"  [{g.get('index')}] {g.get('name')}  {g.get('total_gb')} GB "
                  f"({g.get('free_gb')} GB free, cc {g.get('capability')})")
        print(f"  torch {tor['version']} / cuda {tor['cuda_compiled']} / bf16={tor['bf16_supported']}")
    else:
        print(f"  no CUDA visible ({tor.get('error') or 'torch.cuda.is_available() == False'})")
        if cli["nvidia-smi"]["present"]:
            print(f"  but nvidia-smi reports: {cli['nvidia-smi'].get('output')}")
            print("  -> torch build likely CPU-only or CUDA/driver mismatch; fix before training.")

    print("\n-- packages --")
    missing_hard, missing_soft = [], []
    for mod, r in pkgs.items():
        mark = "ok " if r["present"] else ("MISS" if r["required"] else "opt ")
        print(f"  [{mark}] {mod:<13} {r['version'] or '':<12} {r['why']}")
        if not r["present"]:
            (missing_hard if r["required"] else missing_soft).append(r["pip"])

    print("\n-- cli tools --")
    for t, r in cli.items():
        print(f"  [{'ok ' if r['present'] else 'MISS'}] {t:<11} {r.get('path') or ''}")
    if not cli["aws"]["present"]:
        missing_hard.append("awscli")

    print("\n" + "=" * 68)
    if missing_hard:
        print("BLOCKING — install before proceeding:")
        print(f"  pip install {' '.join(sorted(set(missing_hard)))}")
    else:
        print("All hard requirements satisfied.")
    if missing_soft:
        print(f"Optional (degraded features only):\n  pip install {' '.join(sorted(set(missing_soft)))}")
    print(f"\nfull report -> {args.out}")
    return 1 if missing_hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
