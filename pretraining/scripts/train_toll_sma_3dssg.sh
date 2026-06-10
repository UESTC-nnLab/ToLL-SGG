#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-pretraining/configs/toll_sma_3dssg.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

torchrun \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  pretraining/main_swav.py \
  --config "$CONFIG"
