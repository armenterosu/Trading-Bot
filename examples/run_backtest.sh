#!/usr/bin/env bash
set -euo pipefail

python src/main.py \
  --mode backtest \
  --config config.yaml \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 1h
