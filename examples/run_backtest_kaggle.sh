#!/usr/bin/env bash
set -euo pipefail

# Load .env if present to get vars
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '
' -0 echo 2>/dev/null || true)
fi

# Defaults can be overridden via env
SYMBOL_ENV=${SYMBOL:-"BTC/USDT"}
TIMEFRAME_ENV=${TIMEFRAME:-"5m"}
STRATEGIES_ENV=${STRATEGIES:-"bollinger, ema_cross, ict, support_resistance"}
# Kaggle specifics (also configurable in config.yaml under backtest.kaggle)
: "${KAGGLE_DATASET:=}"
: "${KAGGLE_FILE:=}"

python src/main.py \
  --mode backtest \
  --config config.yaml \
  --exchange "kaggle" \
  --symbol "${SYMBOL_ENV}" \
  --timeframe "${TIMEFRAME_ENV}" \
  --strategies "${STRATEGIES_ENV}"
