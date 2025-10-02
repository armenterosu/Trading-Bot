#!/usr/bin/env bash
set -euo pipefail

python src/main.py \
  --mode paper \
  --config config.yaml \
  --exchange pepperstone \
  --symbol "EUR/USD" \
  --timeframe 15m \
  --dry-run true
