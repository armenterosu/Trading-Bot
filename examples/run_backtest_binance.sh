#!/usr/bin/env bash
set -euo pipefail

# Load .env if present to get vars
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '
' -0 echo 2>/dev/null || true)
fi


python src/main.py \
  --mode backtest \
  --config config.yaml \
  --exchange "binance" \
  --symbol "${SYMBOL_BINANCE}" \
  --timeframe "1h"
