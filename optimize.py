"""Parameter optimization with joblib parallelism.

Usage:
  python optimize.py --config config.yaml --exchange binance --symbol BTC/USDT --timeframe 1h \
    --grid '{"ema_cross.fast": [10,20,30], "ema_cross.slow": [30,50,100]}'
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Any, List
from copy import deepcopy
from joblib import Parallel, delayed

from src.app.utils.config_loader import load_config
from src.app.backtester.backtest import Backtester


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--exchange", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--grid", required=True, help="JSON dict of param grid, e.g. '{""ema_cross.fast"": [10,20]}'")
    p.add_argument("--n_jobs", type=int, default=-1)
    return p.parse_args()


def set_param(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split('.')
    cur = cfg
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    # Cartesian product like sklearn ParameterGrid (simplified)
    items = list(grid.items())
    if not items:
        return [{}]
    key, vals = items[0]
    rest = dict(items[1:])
    combos_rest = product(rest)
    out = []
    for v in vals:
        for r in combos_rest:
            d = dict(r)
            d[key] = v
            out.append(d)
    return out


def eval_combo(base_cfg: Dict[str, Any], params: Dict[str, Any], symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    for dotted, val in params.items():
        set_param(cfg, dotted, val)
    bt = Backtester(cfg)
    res = bt.run(symbol=symbol, exchange=exchange, timeframe=timeframe)
    res_out = {"params": params, "FinalEquity": res.get("FinalEquity", 0.0), "CAGR": res.get("CAGR", 0.0), "Sharpe": res.get("Sharpe", 0.0), "Calmar": res.get("Calmar", 0.0)}
    return res_out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    grid = json.loads(args.grid)
    combos = product(grid)
    results = Parallel(n_jobs=args.n_jobs, prefer="processes")(delayed(eval_combo)(cfg, c, args.symbol, args.exchange, args.timeframe) for c in combos)
    results_sorted = sorted(results, key=lambda x: (-x["FinalEquity"], -x["CAGR"]))
    print(json.dumps(results_sorted[:10], indent=2))


if __name__ == "__main__":
    main()
