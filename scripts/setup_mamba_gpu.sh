#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="reinforcement3-pipcuda"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CHECK='import torch; print("torch:", torch.__version__); print("torch CUDA build:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable; do not start training"; print("GPU:", torch.cuda.get_device_name(0)); print("CUDA tensor check:", torch.ones(3, device="cuda").sum().item())'

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  echo "Installing CUDA PyTorch into active environment: ${CONDA_PREFIX:-$ENV_NAME}"
  python -m pip install -r requirements-gpu.txt --timeout 600 --retries 8
  python -c "$CHECK"
elif command -v mamba >/dev/null 2>&1; then
  mamba env create --strict-channel-priority -f environment.yml
  mamba run -n "$ENV_NAME" python -m pip install -r requirements-gpu.txt --timeout 600 --retries 8
  mamba run -n "$ENV_NAME" python -c "$CHECK"
else
  echo "mamba is not on PATH. If the environment is already active, run the CUDA check directly:" >&2
  echo "python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'" >&2
  echo "Otherwise initialize your Mamba shell and create it with:" >&2
  echo "mamba env create --strict-channel-priority -f environment.yml" >&2
  exit 1
fi
