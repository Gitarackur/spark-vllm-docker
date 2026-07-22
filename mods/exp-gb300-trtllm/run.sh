#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="[exp-gb300-trtllm]"

if ! command -v python3 >/dev/null 2>&1; then
  echo "$PREFIX python3 is required to apply this mod." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "$PREFIX vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_vllm.py" "$PYTHON_ROOT"

# Do not leave bytecode compiled from the pre-patch source around. The patcher
# syntax-checks the result, so clearing the generated cache is safe.
find "$PYTHON_ROOT/vllm/model_executor/model_loader" \
  -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX Enabled GB300 HBM staging for TRTLLM NVFP4 MoE conversion."
