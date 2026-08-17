#!/usr/bin/env bash
# Create the dedicated TACTUS conda env. Idempotent-ish: safe to re-run.
set -uo pipefail
source /home/infres/yinwang/anaconda3/etc/profile.d/conda.sh

ENV_NAME=tactus
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -n "$ENV_NAME" python=3.11 -y -c conda-forge || exit 1
fi
conda activate "$ENV_NAME"
python -V

PIP="python -m pip"
$PIP install --upgrade pip wheel setuptools

# --- torch: cu124 wheels (cluster has cuda 12.x modules; driver >= 550 assumed)
$PIP install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# --- scientific stack
$PIP install \
  "numpy>=1.26,<2.2" "scipy>=1.11" "pandas>=2.1" "pyarrow>=15" "scikit-learn>=1.4" \
  statsmodels joblib numexpr h5py

# --- EEG
$PIP install "mne>=1.7" autoreject python-picard

# --- config / cli
$PIP install "omegaconf>=2.3" pyyaml tqdm

# --- video
$PIP install opencv-python-headless av pillow "transformers>=4.44" safetensors accelerate einops timm

# --- plotting / dev
$PIP install "matplotlib>=3.8" seaborn pytest ruff

# --- data transfer
$PIP install awscli boto3

# --- EEG foundation-model baselines (ladder rung 5: LaBraM / CBraMod / EEGPT)
# braindecode ships all three architectures AND their pretrained checkpoints are
# mirrored on the HF hub (braindecode/{labram,cbramod,eegpt}-pretrained), which
# avoids reimplementing three papers.
# ORDER MATTERS: braindecode pulls torchaudio, and pip resolves it to the latest
# build (2.11.x), which links libcudart.so.13 and dies on import against our
# torch 2.6.0+cu124. Reinstall the matching torchaudio immediately afterwards.
$PIP install "braindecode==1.7.0"
$PIP install "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124

python - <<'EOF'
import importlib
mods = ["torch","torchvision","torchaudio","braindecode","mne","autoreject","picard","numpy","scipy","pandas","pyarrow",
        "sklearn","statsmodels","joblib","h5py","omegaconf","yaml","tqdm","cv2","av","PIL",
        "transformers","safetensors","accelerate","einops","timm","matplotlib","seaborn","pytest"]
bad=[]
for m in mods:
    try:
        mod=importlib.import_module(m); print(f"  ok  {m:<14} {getattr(mod,'__version__','?')}")
    except Exception as e:
        bad.append(m); print(f"  !!  {m:<14} {e}")
print("MISSING:", bad if bad else "none")
EOF
echo "=== setup_env done ==="
