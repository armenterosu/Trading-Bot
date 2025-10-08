#!/usr/bin/env python3
"""Quick test script to debug backtest metrics."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app.backtester.backtrader_engine import BacktraderBacktester
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override for testing
config['backtest'] = config.get('backtest', {})
config['backtest']['limit'] = 100  # Small dataset for quick test

# Create backtester
bt = BacktraderBacktester(config)

# Run with a known symbol
symbol = os.environ.get('SYMBOL', 'BTC/USDT')
print(f"Testing backtest with {symbol} on 1d timeframe...")

try:
    results = bt.run(symbol=symbol, exchange='binance', timeframe='1d', strategies=None)
    print("\n=== RESULTS ===")
    for strat_name, metrics in results.items():
        print(f"\nStrategy: {strat_name}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
