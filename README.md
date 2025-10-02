# Trading Bot (Binance + Pepperstone/cTrader Open API)

Production-ready, modular Python trading bot for multi-market trading:
- Crypto via Binance (ccxt)
- FX/CFDs via Pepperstone (cTrader Open API adapter)

Includes: adapters/connectors, multiple strategies, indicators, backtester, paper/live engine, logging/metrics, Docker, CI, and tests.

WARNING & DISCLAIMER
- This software is for educational and research purposes. Trading involves risk of loss. Use at your own risk.
- Do not commit real API keys. Use environment variables and `.env`.
- Default mode is dry-run for any live adapter.

## Features
- Clean architecture with adapters for brokers (`ccxt` for Binance; cTrader Open API for Pepperstone).
- Strategy framework with plugins: Bollinger, EMA cross, Support/Resistance, ATR trailing stop, basic ICT primitives.
- Backtester with slippage, commissions, stop handling, and performance metrics (CAGR, Sharpe, MaxDD, DD Duration, Calmar, WinRate, Profit Factor, Expectancy, Trades).
- Walk-forward/cross-validation basic utilities.
- Paper trading using the same adapter interface.
- Structured logging (JSON-friendly), rotating logs, CSV metrics export, and optional local HTTP metrics endpoint.
- Dockerized runtime, docker-compose for local stack.
- GitHub Actions CI: lint + tests.

## Project Structure
```
trading-bot/
├─ README.md
├─ LICENSE
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ config.yaml
├─ EXAMPLE_RUNS.md
├─ .env.example
├─ .gitignore
├─ src/
│  ├─ main.py
│  ├─ app/
│  │  ├─ core/
│  │  │  ├─ engine.py
│  │  ├─ exchanges/
│  │  │  ├─ base_adapter.py
│  │  │  ├─ binance_adapter.py
│  │  │  ├─ pepperstone_adapter.py
│  │  ├─ strategies/
│  │  │  ├─ base_strategy.py
│  │  │  ├─ bollinger.py
│  │  │  ├─ ema_cross.py
│  │  │  ├─ support_resistance.py
│  │  │  ├─ ict.py
│  │  ├─ indicators/
│  │  │  ├─ indicators.py
│  │  ├─ risk/
│  │  │  ├─ position_sizing.py
│  │  ├─ backtester/
│  │  │  ├─ backtest.py
│  │  ├─ utils/
│  │  │  ├─ metrics.py
│  │  │  ├─ logging_config.py
│  │  │  ├─ config_loader.py
│  │  │  ├─ kill_switch.py
│  │  │  ├─ metrics_server.py
├─ tests/
│  ├─ test_indicators.py
│  ├─ test_strategies.py
│  ├─ test_backtester.py
├─ .github/workflows/ci.yml
├─ examples/
│  ├─ run_backtest.sh
│  ├─ run_papertrade.sh
│  ├─ run_live.sh
```

## Setup
1) Python 3.10+
2) Create and activate venv
```bash
python -m venv .venv
source .venv/bin/activate
```
3) Install dependencies
```bash
pip install -r requirements.txt
```
4) Configure settings in `config.yaml` (or via env). Copy `.env.example` to `.env` and fill values.

## Docker
Build and run:
```bash
docker compose up --build
```
This starts the bot in paper/live mode as configured, plus optional services.

## Configuration
See `config.yaml` for:
- `global`: timezone, base currency, starting capital
- `exchanges`: Binance and Pepperstone sections
- `indicators`: defaults
- `strategies`: list of enabled strategies and their parameters
- `risk`: risk per trade, limits, drawdown circuit breaker
- `logging`: level, file path, rotation
- `backtest`: data path, dates, commission, slippage

Environment variables override config when present (e.g., `${VAR}` placeholders). `.env` is loaded automatically.

## Usage
- Backtest BTC/USDT short sample:
```bash
python src/main.py --mode backtest --config config.yaml --symbol BTC/USDT --exchange binance
```
- Papertrade EUR/USD with Pepperstone (cTrader Open API adapter):
```bash
python src/main.py --mode paper --config config.yaml --symbol EUR/USD --exchange pepperstone
```
- Live: dry-run is enabled by default to avoid sending real orders. To send real orders, set `--dry-run false` (at your own risk):
```bash
python src/main.py --mode live --config config.yaml --symbol BTC/USDT --exchange binance --dry-run false
```

See `EXAMPLE_RUNS.md` for more.

## Pepperstone Connectivity
- cTrader Open API: uses `ctrader-open-api` (see `requirements.txt`). Configure credentials under `exchanges.pepperstone.ctrader` in `config.yaml`:
  - `access_token`, `account_id`, `environment`, `client_id`, `client_secret` (can be set via `.env`).
  The adapter implements a common interface: `connect`, `get_ohlcv`, order placement/modification/cancel, balances and positions.

## Metrics & Logging
- Logs: structured, rotating file. Configure via `config.yaml`.
- Metrics: CSV export to `./metrics` and a simple local HTTP endpoint (Flask) if enabled, at `http://0.0.0.0:8000/metrics`.

## Tests and CI
Run tests:
```
pytest -q
```
CI runs flake8, mypy (basic), and pytest.

## Security & Ops
- Do not hardcode keys. Use env vars or `.env`.
- Built-in circuit breaker: kill-switch if max drawdown exceeded.
- Rate limiting and retries handled in adapters with backoff.

## Optimization
- Vectorized pandas operations where practical.
- `optimize.py` script parallelizes parameter grid search with joblib and produces a CSV/HTML report.

## Regulatory Notes
- Ensure compliance with local regulations and broker terms. Past results do not guarantee future performance.
