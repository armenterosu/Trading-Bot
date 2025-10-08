"""
Data loader for local historical data files.
"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import glob
from loguru import logger

class DataLoader:
    """Loads historical price data from local files."""
    
    def __init__(self, base_dir: str = 'data'):
        """Initialize with base directory containing symbol subdirectories."""
        self.base_dir = Path(base_dir)
        
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols with downloaded data."""
        return [d.name for d in self.base_dir.glob('*') if d.is_dir()]
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """Get list of available timeframes for a symbol."""
        symbol_dir = self.base_dir / symbol
        if not symbol_dir.exists():
            return []
        return [d.name for d in symbol_dir.glob('*') if d.is_dir()]
    
    def load_historical_data(
        self, 
        symbol: str, 
        timeframe: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """
        Load historical data for a symbol and timeframe.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT' or 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Clean symbol (remove / if present)
        clean_symbol = symbol.replace('/', '')
        data_dir = self.base_dir / clean_symbol / timeframe
        
        if not data_dir.exists():
            raise FileNotFoundError(f"No data directory found for {symbol} {timeframe}")
            
        # Find all CSV files for this symbol and timeframe
        # Look for both naming patterns: {symbol}_{timeframe}_*.csv and {symbol}_{timeframe}*.csv
        files = []
        patterns = [
            f"{clean_symbol}_{timeframe}_*.csv",
            f"{clean_symbol}_{timeframe}*.csv"
        ]
        
        for pattern in patterns:
            matched_files = sorted(glob.glob(str(data_dir / pattern)))
            if matched_files:
                files = matched_files
                break
        
        if not files:
            raise FileNotFoundError(f"No data files found for {symbol} {timeframe}")
            
        # Load and concatenate all data
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file}: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        # Combine all data
        combined = pd.concat(dfs).sort_index()
        
        # Remove duplicates (in case of overlapping files)
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # Filter by date range if specified
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            combined = combined[combined.index >= start_date]
            
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            combined = combined[combined.index <= end_date]
            
        return combined

def load_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    base_dir: str = 'data'
) -> pd.DataFrame:
    """
    Convenience function to load data for a symbol and timeframe.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '1h', '4h', '1d')
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        base_dir: Base directory containing data files
        
    Returns:
        DataFrame with OHLCV data
    """
    loader = DataLoader(base_dir=base_dir)
    return loader.load_historical_data(symbol, timeframe, start_date, end_date)
