#!/usr/bin/env bash
set -euo pipefail

docker compose exec -T swap_manager python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print("capability:", capability)
    print("compiled arch list:", arch_list)
    if capability >= (12, 0) and "sm_120" not in arch_list:
        raise SystemExit("PyTorch sees RTX 50xx but was built without sm_120 support")
else:
    raise SystemExit("CUDA is not available inside swap_manager")
PY
