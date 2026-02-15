import requests
import time
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path
from .utils import save_data

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(
    symbol: str,
    interval: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetches historical klines (candlesticks) from Binance API with pagination.
    
    Downloads OHLCV data from Binance's public API, automatically handling
    pagination to retrieve data beyond the 1000-candle API limit. Requests
    are made sequentially with rate limiting to avoid API restrictions.
    
    The function ensures complete data coverage between start_time and end_time
    by making multiple requests if necessary, seamlessly stitching the results
    together and removing any duplicate timestamps.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT').
        Case-insensitive, will be converted to uppercase.
    interval : str
        Candlestick interval/timeframe. Valid values include:
        - Minutes: '1m', '3m', '5m', '15m', '30m'
        - Hours: '1h', '2h', '4h', '6h', '8h', '12h'
        - Days: '1d', '3d'
        - Weeks: '1w'
        - Months: '1M'
    start_time : str, optional
        Start date in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).
        Required for pagination. If not provided, raises ValueError.
    end_time : str, optional
        End date in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).
        If not provided, defaults to current time.
    limit : int, default 1000
        Maximum number of candles to request per API call.
        Binance API maximum is 1000. Lower values may be used for testing.
        
    Returns
    -------
    pd.DataFrame
        Normalized OHLCV time series with columns:
        - Index: 'open_time' (DatetimeIndex) - Candle opening timestamp
        - 'open': Opening price (float)
        - 'high': Highest price (float)
        - 'low': Lowest price (float)
        - 'close': Closing price (float)
        - 'volume': Trading volume (float)  
    """
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    
    if start_time:
        start_ts = int(datetime.fromisoformat(start_time).timestamp() * 1000)
    else:
        raise ValueError("start_time must be provided for paginated download")
    
    if end_time:
        end_ts = int(datetime.fromisoformat(end_time).timestamp() * 1000)
    else:
        end_ts = int(time.time() * 1000)
        
    all_data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        params["startTime"] = current_ts
        params["endTime"] = end_ts
        response = requests.get(BINANCE_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
        
        all_data.extend(data)
        last_open_time = data[-1][0]
        next_ts = last_open_time + 1
        
        if next_ts <= current_ts:
            break
        
        current_ts = next_ts
        time.sleep(0.1)
    
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]
    
    df = pd.DataFrame(all_data, columns=columns)
    
    df = df[[
        "open_time", "open", "high", "low", "close", "volume"
    ]]
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="first")]
    
    return df


def validate_time_series(df: pd.DataFrame) -> None:
    """
    Performs integrity checks on OHLCV time series data.
    
    Validates that a DataFrame meets the requirements for safe use in backtesting
    and live trading. Checks for common data quality issues including empty data,
    unsorted timestamps, duplicates, and invalid price/volume values.
    
    This function should be called after loading data and before passing it to
    any trading engine or analysis component to prevent runtime errors and
    ensure data integrity.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV time series with DatetimeIndex and numeric columns.
        Expected to have been processed by fetch_klines() or equivalent.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted")
    
    if df.index.has_duplicates:
        raise ValueError("Duplicated timestamps detected")
    
    if (df <= 0).any().any():
        raise ValueError("Non-positive values detected in OHLCV data")
    