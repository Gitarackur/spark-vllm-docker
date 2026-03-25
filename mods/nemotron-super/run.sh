#!/bin/bash
set -e
# cd $WORKSPACE_DIR
# wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/resolve/main/super_v3_reasoning_parser.py



# MY MOD OF RHE MODS (from resolution error on wget downloading it directly from huggingface)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILE="$SCRIPT_DIR/super_v3_reasoning_parser.py"

if [[ ! -f "$FILE" ]]; then
   echo "File not found: $FILE"
   exit 1
fi

cp "$FILE" "$WORKSPACE_DIR/"


