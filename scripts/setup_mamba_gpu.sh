#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="reinforcement3-gpu"
mamba env create -f environment.yml
mamba run -n "$ENV_NAME" python -m pip install --upgrade pip
mamba run -n "$ENV_NAME" python -m pip install -r requirements-gpu.txt --timeout 600 --retries 10
mamba run -n "$ENV_NAME" python -c 'import torch; print("torch:", torch.__version__); print("torch CUDA build:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); [print(f"GPU {i}:", torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]; raise SystemExit(0 if torch.cuda.is_available() else "CUDA unavailable; do not start training")'
