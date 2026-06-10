#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-finetuning/configs/mmgnet.json}"

python finetuning/main.py \
  --no_ddp \
  --config "$CONFIG"
