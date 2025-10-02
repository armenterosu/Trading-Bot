#!/usr/bin/env bash
set -euo pipefail

# Load .env if present to get SYMBOL and other vars
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '\n' -0 echo 2>/dev/null || true)
fi

SYMBOL_ENV=${SYMBOL:-"EUR/USD"}
EXCHANGE_ENV=${EXCHANGE:-"pepperstone"}
TIMEFRAME_ENV=${TIMEFRAME:-"15m"}
DRYRUN_ENV=${DRY_RUN:-"true"}

python src/main.py \
  --mode paper \
  --config config.yaml \
  --exchange "${EXCHANGE_ENV}" \
  --symbol "${SYMBOL_ENV}" \
  --timeframe "${TIMEFRAME_ENV}" \
  --dry-run "${DRYRUN_ENV}"
