import pandas as pd
import numpy as np
from pathlib import Path
from .utils import load_data, save_data

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates exponential and simple moving averages on closing prices.
    
    Adds three moving average columns to the DataFrame:
    - EMA(20): Fast exponential moving average for short-term trend
    - EMA(50): Slow exponential moving average for long-term trend
    - SMA(20): Simple moving average for comparison/crossover strategies
    
    Moving averages are commonly used for trend identification, support/resistance
    levels, and generating trading signals via crossovers.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV time series with 'close' column
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - 'ema_20': 20-period exponential moving average
        - 'ema_50': 50-period exponential moving average
        - 'sma_20': 20-period simple moving average
    """
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int=14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI) momentum oscillator.
    
    RSI measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions. Values range from 0 to 100, with traditional
    thresholds at 30 (oversold) and 70 (overbought).
    
    The implementation uses the standard Wilder's smoothing method with
    rolling averages of gains and losses.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV time series with 'close' column
    period : int, default 14
        Number of periods for RSI calculation (14 is standard)
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added column:
        - 'rsi_14': 14-period RSI values (0-100 range)
    """
    delta = df["close"].diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    return df

def add_atr(df: pd.DataFrame, period: int=14) -> pd.DataFrame:
    """
    Calculates the Average True Range (ATR) volatility indicator.
    
    ATR measures market volatility by calculating the average of true ranges
    over a specified period. It's widely used for position sizing, setting
    stop-loss levels, and gauging market volatility.
    
    True Range is the greatest of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV time series with 'high', 'low', and 'close' columns
    period : int, default 14
        Number of periods for ATR calculation (14 is standard)
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added column:
        - 'atr_14': 14-period Average True Range in price units
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=period).mean()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates logarithmic returns from closing prices.
    
    Logarithmic returns have mathematical properties that make them preferable 
    for statistical analysis: they're time-additive, normally distributed
    (under certain assumptions), and symmetric for gains/losses.
    
    Formula: log_return = ln(price_t / price_t-1)
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV time series with 'close' column
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added column:
        - 'log_return': Period-over-period logarithmic returns
    """
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates volume-based features for analyzing trading activity.
    
    Adds volume moving average and volume ratio to identify unusual trading
    activity. High volume often confirms trend strength or signals potential
    reversals, while low volume suggests weak conviction.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV time series with 'volume' column
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - 'volume_ma_20': 20-period simple moving average of volume
        - 'volume_ratio': Current volume / 20-period average volume
    """
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
    return df

def build_features(
    df: pd.DataFrame,
    symbol: str="ETHUSDT", 
    interval: str="1h", 
    is_backtest: bool=True
) -> pd.DataFrame:
    """
    Builds a complete feature set from raw OHLCV data.
    
    Orchestrates the feature engineering pipeline by sequentially applying
    all technical indicator calculations. This creates a comprehensive dataset
    ready for strategy evaluation, backtesting, or machine learning.
    
    The function applies indicators in a specific order and handles missing
    values appropriately for backtesting vs. live trading contexts.
    
    Features added:
    - Moving averages: EMA(20), EMA(50), SMA(20)
    - Momentum: RSI(14)
    - Volatility: ATR(14)
    - Returns: Logarithmic returns
    - Volume: VMA(20), volume ratio
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV time series (output from fetch_klines or similar)
    symbol : str, default "ETHUSDT"
        Trading pair symbol (used for saving processed data)
    interval : str, default "1h"
        Timeframe interval (used for saving processed data)
    is_backtest : bool, default True
        If True: removes NaN rows and saves processed data to disk
        If False: keeps NaN rows (for live trading where history builds gradually)
        
    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with all technical indicators added.
        If is_backtest=True: NaN rows removed, ready for backtesting.
        If is_backtest=False: All rows retained, suitable for live updates.
    """
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_returns(df)
    df = add_volume_features(df)
    
    if is_backtest:
        df = df.dropna()
        save_data(df, "processed", f"{symbol}_{interval}_PROCESSED.csv")

    return df