# Example Runs

- Backtest BTC/USDT with EMA Cross + Bollinger
```
python src/main.py --mode backtest --config config.yaml --exchange binance --symbol BTC/USDT --timeframe 1h
```

- Paper trade EUR/USD on Pepperstone (REST-sim adapter)
```
python src/main.py --mode paper --config config.yaml --exchange pepperstone --symbol "EUR/USD" --timeframe 15m
```

- Optimize EMA periods (joblib parallel)
```
python optimize.py --config config.yaml --exchange binance --symbol BTC/USDT --timeframe 1h \
  --grid '{"ema_cross.fast": [10,20,30], "ema_cross.slow": [30,50,100]}'
```

- Start metrics HTTP endpoint (optional, enabled by config)
```
python src/main.py --mode backtest --config config.yaml --metrics true
```
