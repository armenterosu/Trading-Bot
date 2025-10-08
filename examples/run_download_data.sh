#!/usr/bin/env bash
set -euo pipefail

# Load .env if present to get vars
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -d '
' -0 echo 2>/dev/null || true)
fi


python -m src.app.utils.data_downloader BTC/USDT ETH/USDT --timeframes 1d 4h 1h