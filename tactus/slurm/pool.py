#!/usr/bin/env python
"""Task-pool runner: run N tasks through W slurm jobs, W << N.

Why this exists
---------------
This cluster enforces ``QOSMaxSubmitJobPerUserLimit = 30`` **counted in array
elements**, so the natural ``--array=1-80`` for the 80-subject preprocess (or
``0-39`` for the double-disjoint training cell, on top of everything else in
flight) is rejected at submit time.  Chunked resubmission would work but leaves
the tail of every chunk idle while the slowest element finishes.

Instead: submit ``W`` identical *worker* jobs (one small array, W <= the free
slot budget).  Each worker loops --- claim the next unclaimed task from a shared
state directory, run it, repeat --- until the task list is exhausted or its
walltime is nearly up.  Load balancing is automatic (a worker that draws three
fast subjects picks up a fourth), the queue footprint is constant, and a
preempted/requeued worker simply rejoins the pool.

Claiming is a single ``O_CREAT|O_EXCL`` open on a shared filesystem, which is
atomic on both NFS and Lustre.  A crashed worker leaves a stale claim without a
``.done``; ``--reclaim-after`` (default 3 h of no heartbeat) lets a later worker
take it over, and ``--retry-failed`` re-queues tasks that exited non-zero.

Usage
-----
    # submit a pool
    python slurm/pool.py submit --name preprocess --tasks 1-80 --workers 12 \
        --partition cpu-high,CPU --time 08:00:00 --cpus 8 --mem 64G \
        --cmd 'python -u -m tactus.data.preprocess --subjects {task} --stage both'

    # inspect
    python slurm/pool.py status --name preprocess
    python slurm/pool.py status --name preprocess --failed        # failing task ids
    python slurm/pool.py reset  --name preprocess --retry-failed  # requeue failures

``{task}`` in ``--cmd`` is substituted with the task id.  ``{task02}`` gives the
zero-padded two-digit form (``sub-07``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "generated"

#: Hard cluster limit on simultaneously-submitted jobs, counted in array
#: elements (probed 2026-08-16: ``--array=1-29`` accepted, ``1-30`` rejected
#: with one job already running).
QOS_MAX_SUBMIT = 30
#: Leave room for interactive shells and the odd one-off job.
SLOT_RESERVE = 3
#: Separate, independently-enforced cap on concurrently *allocated GPUs* per
#: user (slurm reason ``QOSMaxGRESPerUser``).  Probed 2026-08-16: with 8
#: ``--gres=gpu:1`` jobs running, every further GPU job pends on that reason
#: while CPU jobs continue to start normally.  A pool sized only against
#: :data:`QOS_MAX_SUBMIT` will therefore happily queue 40 GPU workers that can
#: never exceed 8 running, which is harmless but makes the queue unreadable and
#: starves later stages behind a pileup.
QOS_MAX_GPUS = 8


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_tasks(spec: str) -> list[int]:
    """``'1-80'`` / ``'1,2,5-9'`` -> a sorted list of ints."""
    out: set[int] = set()
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def free_slots(gpus_per_worker: int = 0) -> int:
    """How many workers we may still usefully submit.

    Bounded by BOTH quotas: the job-count cap and, for GPU pools, the
    concurrent-GPU cap.  Returning the min keeps the queue honest -- submitting
    more GPU workers than :data:`QOS_MAX_GPUS` does not make anything run
    sooner, it just buries the queue.
    """
    user = os.environ.get("USER", "")
    used_jobs, used_gpus = 0, 0
    try:
        r = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i"],
                           capture_output=True, text=True, timeout=30)
        used_jobs = len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        pass
    by_jobs = QOS_MAX_SUBMIT - SLOT_RESERVE - used_jobs
    if gpus_per_worker <= 0:
        return max(1, by_jobs)
    try:
        g = subprocess.run(["squeue", "-h", "-u", user, "-t", "R", "-o", "%b"],
                           capture_output=True, text=True, timeout=30)
        for line in g.stdout.splitlines():
            line = line.strip()
            if "gpu" in line:
                used_gpus += int(line.rsplit(":", 1)[-1]) if line.rsplit(":", 1)[-1].isdigit() else 1
    except Exception:
        pass
    by_gpus = (QOS_MAX_GPUS - used_gpus) // max(1, gpus_per_worker)
    return max(1, min(by_jobs, by_gpus))


class PoolDirs:
    def __init__(self, root: Path, name: str) -> None:
        self.base = Path(root) / "pool" / name
        self.claims = self.base / "claims"
        self.done = self.base / "done"
        self.fail = self.base / "fail"
        self.logs = self.base / "logs"
        self.spec = self.base / "pool.json"

    def mkdirs(self) -> None:
        for d in (self.base, self.claims, self.done, self.fail, self.logs):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def _deadline(max_seconds: float | None) -> float:
    """Epoch second after which the worker stops claiming new tasks."""
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end:
        try:
            return float(end)
        except ValueError:
            pass
    return time.time() + (max_seconds if max_seconds else 3600.0 * 24)


def substitute(template: str, task: int) -> str:
    """Substitute ``{task}``/``{task02}``/``{task03}`` only.

    ``str.format`` is not usable here: commands routinely contain shell syntax
    with braces (``${ARR[0]}``, ``${VAR:-default}``) that ``format`` tries to
    interpret as fields and dies on (``KeyError: 'M'``).
    """
    for key, val in (("{task02}", f"{task:02d}"), ("{task03}", f"{task:03d}"),
                     ("{task}", str(task))):
        template = template.replace(key, val)
    return template


def _heartbeat(path: Path, worker: str) -> None:
    """(Re)write a claim file with a fresh wall-clock heartbeat."""
    path.write_text(json.dumps({
        "worker": worker, "host": socket.gethostname(),
        "job": os.environ.get("SLURM_JOB_ID", "-"), "pid": os.getpid(),
        "heartbeat": time.time(), "heartbeat_iso": _now(),
    }))


def _claim_age(path: Path) -> float | None:
    """Seconds since the claim's last heartbeat, or ``None`` if undecidable.

    The heartbeat is stored **inside** the file rather than read from
    ``st_mtime``: this shared filesystem returned ``st_mtime == 0`` for
    just-created files, which made every worker think a one-second-old claim was
    53 years stale and steal it.  An unreadable or half-written claim returns
    ``None`` and is treated as live -- running a task twice is worse than
    leaving one for the next pool submission.
    """
    try:
        rec = json.loads(path.read_text())
        hb = float(rec["heartbeat"])
    except Exception:
        return None
    return time.time() - hb


def _claim(dirs: PoolDirs, task: int, worker: str, reclaim_after: float) -> bool:
    """Atomically claim ``task``.  Returns False if someone else holds it."""
    if (dirs.done / f"{task}.done").exists():
        return False
    path = dirs.claims / f"{task}.claim"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        _heartbeat(path, worker)
        return True
    except FileExistsError:
        # stale-claim takeover: no heartbeat for reclaim_after seconds and no .done
        age = _claim_age(path)
        if age is None or age <= reclaim_after:
            return False
        if (dirs.done / f"{task}.done").exists():
            return False
        try:
            _heartbeat(path, worker)              # steal it, refresh the heartbeat
            print(f"  [pool] reclaimed stale task {task} (idle {age / 3600:.1f} h)")
            return True
        except OSError:
            return False


def run_worker(args: argparse.Namespace) -> int:
    dirs = PoolDirs(Path(args.state_root), args.name)
    dirs.mkdirs()
    spec = json.loads(dirs.spec.read_text())
    tasks: list[int] = spec["tasks"]
    cmd_tmpl: str = spec["cmd"]

    worker = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    deadline = _deadline(args.max_seconds)
    reserve = float(args.task_seconds_estimate)
    n_ran = n_ok = 0

    print(f"[pool:{args.name}] worker={worker} tasks={len(tasks)} "
          f"budget={(deadline - time.time()) / 3600:.2f} h", flush=True)

    while True:
        left = deadline - time.time()
        if left < reserve:
            print(f"[pool:{args.name}] worker={worker} stopping: {left / 60:.1f} min left "
                  f"< {reserve / 60:.1f} min reserve", flush=True)
            break
        nxt = None
        for t in tasks:
            if (dirs.done / f"{t}.done").exists() or (dirs.fail / f"{t}.fail").exists():
                continue
            if _claim(dirs, t, worker, args.reclaim_after):
                nxt = t
                break
        if nxt is None:
            print(f"[pool:{args.name}] worker={worker}: no unclaimed tasks left", flush=True)
            break

        cmd = substitute(cmd_tmpl, nxt)
        log = dirs.logs / f"{nxt}.log"
        print(f"[pool:{args.name}] worker={worker} -> task {nxt}\n    $ {cmd}", flush=True)
        t0 = time.time()
        with log.open("a") as fh:
            fh.write(f"\n===== task {nxt} worker {worker} {_now()} =====\n$ {cmd}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=spec.get("cwd") or None)
            # heartbeat the claim so a long task is not reclaimed underneath us
            claim = dirs.claims / f"{nxt}.claim"
            while proc.poll() is None:
                time.sleep(60)
                try:
                    _heartbeat(claim, worker)
                except OSError:
                    pass
            rc = proc.returncode
        dt = time.time() - t0
        n_ran += 1
        rec = json.dumps({"task": nxt, "rc": rc, "worker": worker, "seconds": round(dt, 1),
                          "finished": _now()})
        if rc == 0:
            (dirs.done / f"{nxt}.done").write_text(rec)
            n_ok += 1
            print(f"[pool:{args.name}] task {nxt} OK in {dt / 60:.1f} min", flush=True)
        else:
            (dirs.fail / f"{nxt}.fail").write_text(rec)
            print(f"[pool:{args.name}] task {nxt} FAILED rc={rc} after {dt / 60:.1f} min "
                  f"(log: {log})", file=sys.stderr, flush=True)

    print(f"[pool:{args.name}] worker={worker} finished: {n_ok}/{n_ran} ok", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
SBATCH_TMPL = """#!/usr/bin/env bash
#SBATCH --job-name=tactus-{name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output={logdir}/%A_%a.out
#SBATCH --error={logdir}/%A_%a.err
#SBATCH --array=0-{last}
{gres}{extra}#SBATCH --requeue
set -uo pipefail
echo "[tactus] pool={name} job=${{SLURM_JOB_ID}} worker=${{SLURM_ARRAY_TASK_ID}} node=$(hostname) start=$(date -Is)"
{env_setup}
cd "{cwd}"
export PYTHONUNBUFFERED=1
export TACTUS_DATA_ROOT="{data_root}"
export TACTUS_BIDS_ROOT="{bids_root}"
export TACTUS_WORK="{work_root}"
export TACTUS_RESULTS_DIR="{work_root}/results"
export HF_HOME="{work_root}/hf_cache"
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -u {pool_py} worker --name {name} --state-root "{state_root}" \\
    --task-seconds-estimate {reserve} --reclaim-after {reclaim}
echo "[tactus] pool={name} worker=${{SLURM_ARRAY_TASK_ID}} done=$(date -Is)"
"""


def cmd_submit(args: argparse.Namespace) -> int:
    tasks = parse_tasks(args.tasks)
    dirs = PoolDirs(Path(args.state_root), args.name)
    dirs.mkdirs()

    pending = [t for t in tasks
               if not (dirs.done / f"{t}.done").exists()
               and not (dirs.fail / f"{t}.fail").exists()]
    dirs.spec.write_text(json.dumps({
        "name": args.name, "tasks": tasks, "cmd": args.cmd, "cwd": args.cwd,
        "created": _now(),
    }, indent=2))

    if not pending:
        print(f"[pool:{args.name}] every task already done/failed; nothing to submit")
        return 0

    budget = free_slots(gpus_per_worker=int(args.gpus or 0))
    workers = max(1, min(args.workers, len(pending), budget))
    if workers < args.workers:
        print(f"[pool:{args.name}] {args.workers} workers requested, {workers} submitted "
              f"(QOS budget {budget}, pending {len(pending)}"
              + (f", GPU cap {QOS_MAX_GPUS}" if args.gpus else "") + ")")

    logdir = Path(args.log_root) / args.name
    logdir.mkdir(parents=True, exist_ok=True)
    script = SBATCH_TMPL.format(
        name=args.name, partition=args.partition, time=args.time, cpus=args.cpus,
        mem=args.mem, logdir=logdir, last=workers - 1,
        gres=f"#SBATCH --gres=gpu:{args.gpus}\n" if args.gpus else "",
        extra="".join(f"#SBATCH {x}\n" for x in args.sbatch),
        env_setup=args.env_setup, cwd=args.cwd, data_root=args.data_root,
        bids_root=args.bids_root, work_root=args.work_root,
        pool_py=str(HERE / "pool.py"), state_root=args.state_root,
        reserve=args.task_seconds_estimate, reclaim=args.reclaim_after,
    )
    GEN.mkdir(parents=True, exist_ok=True)
    path = GEN / f"pool_{args.name}.sbatch"
    path.write_text(script)
    path.chmod(0o755)

    print(f"[pool:{args.name}] {len(pending)} pending task(s), {workers} worker(s)")
    print(f"           partition={args.partition} time={args.time} cpus={args.cpus} mem={args.mem}"
          + (f" gpus={args.gpus}" if args.gpus else ""))
    print(f"           cmd: {args.cmd}")
    print(f"           script={path}")
    if args.dry_run:
        print(f"           would run: sbatch {path}")
        return 0

    cmd = ["sbatch", "--parsable"]
    if args.dependency:
        cmd.append(f"--dependency={args.dependency}")
    cmd.append(str(path))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"sbatch failed:\n{r.stderr}", file=sys.stderr)
        return 1
    jid = r.stdout.strip().split(";")[0]
    print(f"           submitted jobid={jid}  logs={logdir}")
    return 0


# --------------------------------------------------------------------------- #
# status / reset
# --------------------------------------------------------------------------- #
def cmd_status(args: argparse.Namespace) -> int:
    dirs = PoolDirs(Path(args.state_root), args.name)
    if not dirs.spec.exists():
        print(f"no pool named {args.name!r} under {dirs.base}", file=sys.stderr)
        return 2
    spec = json.loads(dirs.spec.read_text())
    tasks = spec["tasks"]
    done = {int(p.stem) for p in dirs.done.glob("*.done")}
    fail = {int(p.stem) for p in dirs.fail.glob("*.fail")}
    claimed = {int(p.stem) for p in dirs.claims.glob("*.claim")}
    running = claimed - done - fail
    todo = [t for t in tasks if t not in done and t not in fail and t not in claimed]

    if args.failed:
        print(",".join(str(t) for t in sorted(fail)))
        return 0
    if args.pending:
        print(",".join(str(t) for t in sorted(set(todo) | running)))
        return 0

    print(f"pool {args.name}: {len(tasks)} tasks")
    print(f"  done    {len(done):>4}")
    print(f"  running {len(running):>4}  {sorted(running)[:20]}")
    print(f"  todo    {len(todo):>4}  {sorted(todo)[:20]}")
    print(f"  failed  {len(fail):>4}  {sorted(fail)[:20]}")
    if fail and args.verbose:
        for t in sorted(fail):
            rec = json.loads((dirs.fail / f"{t}.fail").read_text())
            print(f"    task {t}: rc={rec['rc']}  log={dirs.logs / f'{t}.log'}")
    secs = [json.loads(p.read_text()).get("seconds", 0) for p in dirs.done.glob("*.done")]
    if secs:
        secs.sort()
        print(f"  per-task minutes: median {secs[len(secs) // 2] / 60:.1f}  max {secs[-1] / 60:.1f}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    dirs = PoolDirs(Path(args.state_root), args.name)
    tasks = parse_tasks(args.tasks) if args.tasks else None
    n = 0
    for p in list(dirs.fail.glob("*.fail")) if args.retry_failed else []:
        if tasks is None or int(p.stem) in tasks:
            p.unlink()
            (dirs.claims / f"{p.stem}.claim").unlink(missing_ok=True)
            n += 1
    for p in list(dirs.claims.glob("*.claim")) if args.clear_claims else []:
        if (dirs.done / f"{p.stem}.done").exists():
            continue
        if tasks is None or int(p.stem) in tasks:
            p.unlink()
            n += 1
    for p in list(dirs.done.glob("*.done")) if args.clear_done else []:
        if tasks is None or int(p.stem) in tasks:
            p.unlink()
            (dirs.claims / f"{p.stem}.claim").unlink(missing_ok=True)
            n += 1
    print(f"[pool:{args.name}] reset {n} marker file(s)")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python slurm/pool.py")
    sub = ap.add_subparsers(dest="action", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", required=True)
        p.add_argument("--state-root", default=os.environ.get(
            "TACTUS_WORK", "/projects/EEG-foundation-model/tactus_work"))

    s = sub.add_parser("submit")
    common(s)
    s.add_argument("--tasks", required=True, help="'1-80' or '1,3,5-9'")
    s.add_argument("--cmd", required=True, help="shell command; {task} / {task02} substituted")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--partition", required=True,
                   help="comma-separated slurm partitions. For GPU work use "
                        "A100,L40S,H100 -- V100/P100/A30/3090 are excluded on "
                        "purpose (see slurm/cluster.conf). Those fast partitions "
                        "cap MaxTime at 24 h, so keep --time under it.")
    s.add_argument("--time", default="08:00:00")
    s.add_argument("--cpus", type=int, default=8)
    s.add_argument("--mem", default="64G")
    s.add_argument("--gpus", type=int, default=0)
    s.add_argument("--sbatch", action="append", default=[], help="extra raw #SBATCH lines")
    s.add_argument("--dependency", default=None)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--cwd", default="/home/infres/yinwang/TACTUS/tactus")
    s.add_argument("--env-setup", default=(
        "source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh && conda activate tactus"))
    s.add_argument("--data-root", default="/projects/EEG-foundation-model/tactus_work")
    s.add_argument("--bids-root", default="/projects/EEG-foundation-model/ds005662")
    s.add_argument("--work-root", default="/projects/EEG-foundation-model/tactus_work")
    s.add_argument("--log-root", default="/projects/EEG-foundation-model/tactus_work/logs")
    s.add_argument("--task-seconds-estimate", type=float, default=1800,
                   help="stop claiming when less than this much walltime remains")
    s.add_argument("--reclaim-after", type=float, default=10800,
                   help="seconds without a claim heartbeat before another worker may steal it")

    w = sub.add_parser("worker")
    common(w)
    w.add_argument("--max-seconds", type=float, default=None)
    w.add_argument("--task-seconds-estimate", type=float, default=1800)
    w.add_argument("--reclaim-after", type=float, default=10800)

    t = sub.add_parser("status")
    common(t)
    t.add_argument("--failed", action="store_true", help="print failed task ids only")
    t.add_argument("--pending", action="store_true", help="print unfinished task ids only")
    t.add_argument("--verbose", "-v", action="store_true")

    r = sub.add_parser("reset")
    common(r)
    r.add_argument("--tasks", default=None)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--clear-claims", action="store_true")
    r.add_argument("--clear-done", action="store_true")

    args = ap.parse_args(argv)
    return {"submit": cmd_submit, "worker": run_worker,
            "status": cmd_status, "reset": cmd_reset}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
