#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default to avoid sending real orders
python src/main.py \
  --mode live \
  --config config.yaml \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1h \
  --dry-run true
