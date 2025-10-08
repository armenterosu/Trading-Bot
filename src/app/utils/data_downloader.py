"""
Historical data downloader for multiple exchanges.

Supports:
- Binance (crypto)
- CTrader (forex/CFDs)
"""
import os
import time
import pandas as pd
import ccxt
from datetime import datetime
from loguru import logger
from typing import Dict, List, Optional


# Timeframes supported by Binance
TIMEFRAMES = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',
    '1M': '1M',
    '1y': '1y',
}

def create_data_directories(base_dir: str, symbol: str) -> Dict[str, str]:
    """Create necessary directories for storing data."""
    symbol_dir = os.path.join(base_dir, symbol.replace('/', ''))
    paths = {}
    
    for tf in TIMEFRAMES.values():
        path = os.path.join(symbol_dir, tf)
        os.makedirs(path, exist_ok=True)
        paths[tf] = path
    
    return paths

def fetch_binance_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: Optional[int] = None,
    limit: int = 1000,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Fetch historical OHLCV data from Binance."""
    all_ohlcv = []
    current_since = since
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not ohlcv:
                break
                
            if current_since and ohlcv[0][0] == current_since:
                ohlcv = ohlcv[1:]
                if not ohlcv:
                    break
            
            # Check if we've reached the end date
            last_timestamp = ohlcv[-1][0]
            last_date = pd.to_datetime(last_timestamp, unit='ms')
            
            if end_date and last_date >= end_date:
                # Filter out data points after end_date
                ohlcv = [candle for candle in ohlcv 
                         if pd.to_datetime(candle[0], unit='ms') <= end_date]
                all_ohlcv.extend(ohlcv)
                break
                
            all_ohlcv.extend(ohlcv)
            current_since = last_timestamp + 1
            
            # Sleep to avoid rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break
    
    if not all_ohlcv:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

def save_historical_data(df: pd.DataFrame, symbol: str, timeframe: str, output_dir: str):
    """Save historical data to CSV file."""
    if df.empty:
        logger.warning(f"No data to save for {symbol} {timeframe}")
        return
        
    # Create filename with date range
    start_date = df.index[0].strftime('%Y%m%d')
    end_date = df.index[-1].strftime('%Y%m%d')
    filename = f"{symbol.replace('/', '')}_{timeframe}_{start_date}_{end_date}.csv"
    filepath = os.path.join(output_dir, timeframe, filename)
    
    # Reset index to include datetime as a column
    df.reset_index(inplace=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} candles to {filepath}")

def download_historical_data(
    symbol: str,
    timeframes: List[str] = None,
    since: str = '2017-01-01',
    base_dir: str = 'data',
    end_date: Optional[datetime] = None,
):
    """Download historical data for a symbol and timeframes."""
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())
    
    # Create necessary directories
    paths = create_data_directories(base_dir, symbol)
    
    # Convert since to datetime
    since_dt = pd.to_datetime(since)
    

    # Initialize Binance exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    since_ts = int(since_dt.timestamp() * 1000)

    # Download data for each timeframe
    for tf in timeframes:
        if tf not in TIMEFRAMES:
            logger.warning(f"Unsupported timeframe: {tf}")
            continue

        tf_str = TIMEFRAMES[tf]
        logger.info(f"[Binance] Downloading {symbol} {tf_str} data since {since_dt}")

        df = fetch_binance_data(exchange, symbol, tf_str, since_ts, end_date=end_date)
        if not df.empty:
            save_historical_data(df, symbol, tf_str, os.path.join(base_dir, symbol.replace('/', '')))
        else:
            logger.warning(f"No data found for {symbol} {tf_str}")




def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download historical data from Binance ')
    parser.add_argument('symbols', type=str, default='BTC/USDT' ,nargs='+', help='Trading pairs (e.g., BTC/USDT)')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h', '4h', '1d'],
                       help='Timeframes to download (1h, 4h, 1d)')
    parser.add_argument('--since', type=str, default='2010-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', type=str, default=None,
                       help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--base-dir', type=str, default='data',
                       help='Base directory to save data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Set end date to today if not specified
    end_date = pd.to_datetime(args.to) if args.to else datetime.now()
    
    for symbol in args.symbols:
        try:
            download_historical_data(
                symbol=symbol,
                timeframes=args.timeframes,
                since=args.since,
                base_dir=args.base_dir,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"Failed to download data for {symbol}: {e}")

if __name__ == "__main__":
    main()
