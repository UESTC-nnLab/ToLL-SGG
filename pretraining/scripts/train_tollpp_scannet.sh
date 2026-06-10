#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-pretraining/configs/tollpp_scannet.json}"

python pretraining/main_diff.py \
  --no_ddp \
  --config "$CONFIG"
