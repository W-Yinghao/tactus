#!/usr/bin/env python
"""TACTUS slurm orchestrator.

Generates inspectable sbatch scripts into slurm/generated/ and submits them as a
dependency chain. Every stage is idempotent and resumable, so jobs may be
requeued on preemption or timeout without corrupting state.

    bash slurm/cluster_probe.sh > slurm/cluster_report.txt   # then fill cluster.conf
    python slurm/submit.py --chain all --dry-run             # inspect
    python slurm/submit.py --chain all                       # submit
    python slurm/submit.py --stage train --config configs/nice_infonce.yaml
    python slurm/submit.py --status                          # squeue for this project

Stage graph:
    download -> trials -> preprocess -> embed -> {mvpa, train} -> eval
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"
CONF_PATH = HERE / "cluster.conf"
JOB_PREFIX = "tactus"


# --------------------------------------------------------------------------- conf
def load_conf(path: Path = CONF_PATH) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"missing {path}. Run `bash slurm/cluster_probe.sh` and fill it in.")
    conf: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        conf[k.strip()] = v.split("  #")[0].strip().strip('"')
    for req in ("DATA_ROOT", "REPO_ROOT", "WORK_ROOT", "LOG_ROOT"):
        if not conf.get(req):
            sys.exit(f"cluster.conf: {req} must be set")
    return conf


def partition_is_visible(name: str) -> bool:
    try:
        r = subprocess.run(["sinfo", "-h", "-p", name, "-o", "%P"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def partition_pressure(name: str) -> tuple[int, int]:
    """(idle_nodes, pending_jobs) — lower pending / higher idle is better."""
    idle = pend = 0
    try:
        r = subprocess.run(["sinfo", "-h", "-p", name, "-t", "idle", "-o", "%D"],
                           capture_output=True, text=True, timeout=20)
        idle = sum(int(x) for x in r.stdout.split() if x.isdigit())
        q = subprocess.run(["squeue", "-h", "-p", name, "-t", "PD"],
                           capture_output=True, text=True, timeout=20)
        pend = len([l for l in q.stdout.splitlines() if l.strip()])
    except Exception:
        pass
    return idle, pend


def pick_partition(candidates: str, probe: bool = True) -> str:
    """Every visible candidate, least-contended first, as a comma list for sbatch.

    ``--partition=A,B,C`` lets the *scheduler* choose: the job starts in whichever
    partition frees a slot first, instead of us guessing at submit time and then
    sitting in one queue. Order still matters as a tie-break hint, so the list is
    sorted by current idle capacity.

    The walltime in ``cluster.conf`` must fit the *shortest* limit among the
    listed partitions -- :func:`check_walltime` enforces that.
    """
    names = [c.strip() for c in candidates.split(",") if c.strip()]
    visible = [n for n in names if partition_is_visible(n)] if probe else names
    if not visible:
        print(f"  [warn] none of {names} visible; falling back to {names[0]}")
        return names[0]
    if not probe or len(visible) == 1:
        return ",".join(visible)
    scored = sorted(visible, key=lambda n: (-partition_pressure(n)[0],
                                            partition_pressure(n)[1], names.index(n)))
    return ",".join(scored)


def _to_seconds(tl: str) -> int:
    """Parse a slurm walltime ('D-HH:MM:SS', 'HH:MM:SS', 'MM:SS') into seconds."""
    tl = tl.strip()
    if tl in ("", "UNLIMITED", "INFINITE"):
        return 10 ** 9
    days = 0
    if "-" in tl:
        d, tl = tl.split("-", 1)
        days = int(d)
    parts = [int(x) for x in tl.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def check_walltime(part_list: str, requested: str) -> str:
    """Clamp ``requested`` to the shortest limit among the listed partitions."""
    want = _to_seconds(requested)
    limits: dict[str, int] = {}
    for p in part_list.split(","):
        try:
            r = subprocess.run(["sinfo", "-h", "-p", p, "-o", "%l"],
                               capture_output=True, text=True, timeout=20)
            vals = {_to_seconds(v) for v in r.stdout.split() if v.strip()}
            if vals:
                limits[p] = min(vals)
        except Exception:
            pass
    if not limits:
        return requested
    lo = min(limits.values())
    if want <= lo:
        return requested
    tight = [p for p, v in limits.items() if v == lo]
    h, rem = divmod(lo, 3600)
    m, s = divmod(rem, 60)
    clamped = f"{h:02d}:{m:02d}:{s:02d}"
    print(f"  [walltime] {requested} exceeds the {tight} limit; clamped to {clamped}")
    return clamped


# --------------------------------------------------------------------------- stages
@dataclass
class Stage:
    name: str
    command: str                      # may contain {conf} placeholders and $SLURM_ARRAY_TASK_ID
    partition_key: str                # conf key holding partition or candidate list
    time_key: str
    cpus_key: str
    mem_key: str
    gpus_key: str | None = None
    array: str | None = None          # e.g. "1-80%10" (throttle read from conf)
    after: list[str] = field(default_factory=list)
    note: str = ""


def build_stages(conf: dict[str, str], cfg_file: str, regime: str, n_folds: int,
                 video_model: str) -> dict[str, Stage]:
    py = "python -u"
    bids = f'{conf["DATA_ROOT"]}/ds005662'
    derived = f'{conf["WORK_ROOT"]}/derived'
    return {
        "download": Stage(
            "download",
            # array task 1..80 downloads one subject; task 81 grabs metadata+stimuli+derivatives
            f'{py} -m tactus.data.download --what auto --subject-index $SLURM_ARRAY_TASK_ID '
            f'--dest "{bids}" --max-concurrent-requests 16 --quiet',
            "CPU_PARTITION", "DOWNLOAD_TIME", "DOWNLOAD_CPUS", "DOWNLOAD_MEM",
            array=f'1-81%{conf.get("DOWNLOAD_ARRAY_THROTTLE", "8")}',
            note="80 subject BDFs in parallel + one metadata/stimuli/derivatives task",
        ),
        "trials": Stage(
            "trials",
            f'{py} -m tactus.data.events --bids-root "{bids}" '
            f'--out "{derived}/trials.parquet" --audit-out "{derived}/trials_audit.json"',
            "CPU_PARTITION", "TRIALS_TIME", "TRIALS_CPUS", "TRIALS_MEM",
            after=["download"],
            note="canonical trial table + the Phase-0 sequence x orientation audit",
        ),
        "preprocess": Stage(
            "preprocess",
            f'{py} -m tactus.data.preprocess --subjects $SLURM_ARRAY_TASK_ID --stage both '
            f'--bids-root "{bids}" --trials "{derived}/trials.parquet" '
            f'--out-dir "{derived}/epochs" --windows w0600,wm100_800 '
            f'--filter-n-jobs ${{SLURM_CPUS_PER_TASK:-8}} --autoreject '
            f'--save-continuous-eog-proxy '
            f'--report "{derived}/epochs/preprocess_report_sub$(printf %02d ${{SLURM_ARRAY_TASK_ID}}).json"',
            "CPU_HIGH_PARTITION", "PREPROCESS_TIME", "PREPROCESS_CPUS", "PREPROCESS_MEM",
            array=f'1-80%{conf.get("PREPROCESS_ARRAY_THROTTLE", "10")}',
            after=["trials"],
            note="BDF -> 0-600ms epochs (no baseline) + ICA ocular components, one task per subject",
        ),
        "embed": Stage(
            "embed",
            f'{py} -m tactus.models.video.encode --model {video_model} '
            f'--stim-root "{bids}" --out "{derived}/video_emb"',
            "GPU_SMALL_PARTITIONS", "EMBED_TIME", "EMBED_CPUS", "EMBED_MEM", "EMBED_GPUS",
            after=["download"],
            note="360 conditions through a frozen video encoder; minutes on any card",
        ),
        "mvpa": Stage(
            "mvpa",
            f'{py} -m tactus.baselines.linear_mvpa --subjects $SLURM_ARRAY_TASK_ID '
            f'--cv sequence --window w0600 --n-jobs ${{SLURM_CPUS_PER_TASK:-8}} '
            f'--out "{conf["WORK_ROOT"]}/results/baselines/mvpa"',
            "CPU_HIGH_PARTITION", "MVPA_TIME", "MVPA_CPUS", "MVPA_MEM",
            array=f'1-80%{conf.get("MVPA_ARRAY_THROTTLE", "16")}',
            after=["preprocess"],
            note="companion-paper reproduction; sanity landmarks orientation~60ms, material~110ms",
        ),
        "train": Stage(
            "train",
            f'{py} -m tactus.train.run --config {cfg_file} --regime {regime} '
            f'--folds $SLURM_ARRAY_TASK_ID --out-root "{conf["WORK_ROOT"]}/results/runs"',
            "GPU_SMALL_PARTITIONS", "TRAIN_TIME", "TRAIN_CPUS", "TRAIN_MEM", "TRAIN_GPUS",
            array=f'0-{n_folds - 1}%{conf.get("TRAIN_ARRAY_THROTTLE", "6")}',
            after=["preprocess", "embed"],
            note="one fold per array task",
        ),
        "fm": Stage(
            "fm",
            f'{py} -m tactus.train.run --config configs/labram_ft.yaml --regime {regime} '
            f'--folds $SLURM_ARRAY_TASK_ID --out-root "{conf["WORK_ROOT"]}/results/runs"',
            "GPU_LARGE_PARTITIONS", "FM_TIME", "FM_CPUS", "FM_MEM", "FM_GPUS",
            array="0-2",
            after=["preprocess", "embed"],
            note="EEG foundation-model fine-tuning, 3 folds only (indicative baseline)",
        ),
        "eval": Stage(
            "eval",
            f'{py} -m tactus.train.run --config {cfg_file} --regime {regime} --aggregate-only '
            f'--out-root "{conf["WORK_ROOT"]}/results/runs" && '
            f'{py} -m tactus.eval.report --run-root "{conf["WORK_ROOT"]}/results/runs" '
            f'--config {cfg_file} --out "{conf["WORK_ROOT"]}/results/REPORT.md"',
            "CPU_HIGH_PARTITION", "EVAL_TIME", "EVAL_CPUS", "EVAL_MEM",
            after=["train"],
            note="fold aggregation, video-level permutation nulls, noise ceilings, REPORT.md",
        ),
    }


CHAINS = {
    "all": ["download", "trials", "preprocess", "embed", "mvpa", "train", "eval"],
    "data": ["download", "trials", "preprocess", "embed"],
    "baseline": ["mvpa", "train", "eval"],
}


# --------------------------------------------------------------------------- sbatch
def render(stage: Stage, conf: dict[str, str], probe: bool) -> tuple[Path, str]:
    part = pick_partition(conf.get(stage.partition_key, ""), probe=probe)
    walltime = check_walltime(part, conf[stage.time_key]) if probe else conf[stage.time_key]
    log = f'{conf["LOG_ROOT"]}/{stage.name}'
    arr = "%A_%a" if stage.array else "%j"

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={JOB_PREFIX}-{stage.name}",
        f"#SBATCH --partition={part}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --cpus-per-task={conf[stage.cpus_key]}",
        f"#SBATCH --mem={conf[stage.mem_key]}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --output={log}/{arr}.out",
        f"#SBATCH --error={log}/{arr}.err",
    ]
    if stage.gpus_key and conf.get(stage.gpus_key):
        lines.append(f"#SBATCH --gres=gpu:{conf[stage.gpus_key]}")
    if stage.array:
        lines.append(f"#SBATCH --array={stage.array}")
    for key, flag in (("ACCOUNT", "account"), ("QOS", "qos"),
                      ("MAIL_USER", "mail-user"), ("MAIL_TYPE", "mail-type")):
        if conf.get(key):
            lines.append(f"#SBATCH --{flag}={conf[key]}")
    if conf.get("REQUEUE", "0") == "1":
        lines.append("#SBATCH --requeue")

    body = f"""
set -euo pipefail
echo "[tactus] stage={stage.name} job=${{SLURM_JOB_ID}} task=${{SLURM_ARRAY_TASK_ID:-none}} node=$(hostname) start=$(date -Is)"
mkdir -p "{log}"
{conf.get("ENV_SETUP") or "# ENV_SETUP not set in cluster.conf"}
cd "{conf["REPO_ROOT"]}"
export PYTHONUNBUFFERED=1
export TACTUS_WORK="{conf["WORK_ROOT"]}"
export TACTUS_DATA="{conf["DATA_ROOT"]}"
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true

{stage.command}

echo "[tactus] stage={stage.name} done=$(date -Is)"
""".rstrip() + "\n"

    text = "\n".join(lines) + "\n" + body
    GEN.mkdir(parents=True, exist_ok=True)
    path = GEN / f"{stage.name}.sbatch"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path, part


def sbatch(path: Path, deps: list[str]) -> str:
    cmd = ["sbatch", "--parsable"]
    if deps:
        cmd.append("--dependency=afterok:" + ":".join(deps))
    cmd.append(str(path))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"sbatch failed for {path.name}:\n{r.stderr}")
    return r.stdout.strip().split(";")[0]


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=sorted(CHAINS))
    ap.add_argument("--stage", action="append", help="submit specific stage(s)")
    ap.add_argument("--config", default="configs/nice_infonce.yaml")
    ap.add_argument("--regime", default="within_subject",
                    choices=["within_subject", "loso", "double_disjoint"])
    ap.add_argument("--n-folds", type=int, default=None,
                    help="default: 10 within_subject / 8 loso / 40 double_disjoint")
    ap.add_argument("--video-model", default="siglip2-base")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-probe", action="store_true", help="skip sinfo partition probing")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        os.execvp("squeue", ["squeue", "-u", os.environ.get("USER", ""), "-o",
                             "%.10i %.22j %.10P %.2t %.10M %.6D %R"])

    conf = load_conf()
    n_folds = args.n_folds or {"within_subject": 10, "loso": 8, "double_disjoint": 40}[args.regime]
    stages = build_stages(conf, args.config, args.regime, n_folds, args.video_model)

    wanted = args.stage or (CHAINS[args.chain] if args.chain else None)
    if not wanted:
        sys.exit("pass --chain {all,data,baseline} or --stage NAME")

    print(f"regime={args.regime}  folds={n_folds}  config={args.config}")
    print(f"work={conf['WORK_ROOT']}\n")

    ids: dict[str, str] = {}
    for name in wanted:
        st = stages[name]
        path, part = render(st, conf, probe=not args.no_probe and not args.dry_run)
        deps = [ids[a] for a in st.after if a in ids]
        arr = f"  array={st.array}" if st.array else ""
        print(f"[{name}] partition={part} time={conf[st.time_key]} "
              f"cpus={conf[st.cpus_key]} mem={conf[st.mem_key]}{arr}")
        if st.note:
            print(f"         {st.note}")
        print(f"         script={path}")
        if args.dry_run:
            dep = f" --dependency=afterok:{':'.join(deps)}" if deps else ""
            print(f"         would run: sbatch{dep} {path}\n")
            ids[name] = f"<{name}>"
        else:
            jid = sbatch(path, deps)
            ids[name] = jid
            print(f"         submitted jobid={jid}"
                  + (f" (after {','.join(deps)})" if deps else "") + "\n")

    if not args.dry_run:
        print("submitted:", " ".join(f"{k}={v}" for k, v in ids.items()))
        print("watch:  python slurm/submit.py --status")
        print(f"logs:   {conf['LOG_ROOT']}/<stage>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
