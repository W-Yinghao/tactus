#!/usr/bin/env bash
# TACTUS cluster probe — run on the LOGIN node. Fills in what submit.py needs to know.
# Usage:  bash slurm/cluster_probe.sh | tee slurm/cluster_report.txt
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/projects/EEG-foundation-model}"

hdr() { printf '\n=== %s ===\n' "$1"; }

hdr "identity"
echo "user=$(whoami)  host=$(hostname)  date=$(date -Is)"
echo "DATA_ROOT=$DATA_ROOT"
ls -ld "$DATA_ROOT" 2>&1 | head -1

hdr "slurm accounts / QOS (needed for #SBATCH --account / --qos)"
sacctmgr -nP show assoc user="$(whoami)" format=Account,Partition,QOS,MaxJobs,MaxSubmit 2>&1 | head -40
echo "--- default account ---"
sacctmgr -nP show user "$(whoami)" format=DefaultAccount 2>&1

hdr "partitions: limits and current load"
sinfo -o "%20P %8a %12l %10D %14F %N" 2>&1 | head -40
echo "(%F = allocated/idle/other/total nodes)"

hdr "per-partition node resources (cpus / mem MB / gres)"
for P in CPU cpu-high H100 A100 L40S A40 V100; do
  line=$(sinfo -h -p "$P" -o "%20P %6c %10m %25G %8t %6D" 2>/dev/null | sort -u | head -6)
  if [ -n "$line" ]; then printf '%s\n' "$line"; else echo "  [partition $P not visible to this user]"; fi
done

hdr "max walltime per partition"
sinfo -h -o "%20P %12l" 2>&1 | sort -u

hdr "GRES detail (GPU types actually schedulable)"
sinfo -h -o "%P %G" 2>&1 | sort -u | grep -i gpu | head -20

hdr "current queue pressure (idle GPUs right now)"
for P in H100 A100 L40S A40 V100; do
  idle=$(sinfo -h -p "$P" -t idle -o "%D" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
  pend=$(squeue -h -p "$P" -t PD 2>/dev/null | wc -l)
  echo "  $P: idle_nodes=${idle:-0}  pending_jobs=${pend:-?}"
done

hdr "array job limits"
scontrol show config 2>/dev/null | grep -Ei "MaxArraySize|MaxJobCount|MaxSubmitJobs|DefMemPerCPU|MaxMemPerCPU" | head

hdr "module system"
if command -v module >/dev/null 2>&1; then
  module avail 2>&1 | grep -Ei "cuda|python|anaconda|miniconda|gcc|ffmpeg" | head -25
else
  echo "  no 'module' command (env is probably conda/venv-based)"
fi

hdr "python / conda in PATH on login node"
echo "python: $(command -v python || echo none)  $(python -V 2>&1)"
echo "conda:  $(command -v conda  || echo none)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}  VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "conda envs:"; conda env list 2>/dev/null | head -15

hdr "aws cli (needed for the s3 sync of ds005662)"
command -v aws >/dev/null 2>&1 && aws --version 2>&1 || echo "  MISSING — pip install awscli"

hdr "filesystem"
df -h "$DATA_ROOT" 2>&1 | tail -2
echo "quota (if enforced):"; quota -s 2>/dev/null | head -5 || echo "  n/a"
echo "scratch candidates:"; for d in /scratch /tmp /local "$TMPDIR"; do [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1; done

hdr "NEXT"
cat <<'EOF'
1. Copy the account/qos/partition names above into slurm/cluster.conf
2. Probe the *compute-node* environment (login node GPUs are usually invisible):
     srun -p L40S --gres=gpu:1 -t 00:10:00 --pty python env_probe.py --skip-disk
3. Then: python -m tactus.slurm.submit --chain all --dry-run
EOF
