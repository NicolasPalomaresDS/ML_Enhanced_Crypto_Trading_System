import pandas as pd
from pathlib import Path
from .rules import *
from .utils import load_data, save_data

def evaluate_strategy(row: pd.Series) -> str:
    """
    Evaluates trading strategy and generates signal for a single row.
    
    Implements a trend-following pullback strategy that combines multiple
    technical conditions to identify high-probability long entry opportunities.
    The strategy only generates BUY signals when all conditions align,
    ensuring confluence of multiple factors.
    
    Strategy logic:
    1. Bullish regime: Confirms uptrend (price > EMA50, EMA20 > EMA50)
    2. RSI pullback: RSI in 40-55 zone (healthy dip, not oversold)
    3. Sufficient volatility: ATR >= 0.3% of price (tradeable moves)
    4. Volume confirmation: Volume >= average (validates move)
    
    This multi-condition approach reduces false signals and aligns entries
    with strong trends during temporary weakness.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated technical
        indicators. Must have columns: close, ema_20, ema_50, rsi_14,
        atr_14, volume_ratio
        
    Returns
    -------
    str
        Trading signal:
        - 'BUY': All conditions met, enter long position
        - 'HOLD': Conditions not met, wait or stay in existing position
    """
    if (
        bullish_regime(row) and
        rsi_pullback_bull(row) and
        sufficient_volatility(row) and
        sufficient_volume(row)
    ):
        return "BUY"

    return "HOLD"


def generate_signals(
    df: pd.DataFrame,
    symbol: str="ETHUSDT", 
    interval: str="1h", 
    is_backtest: bool=True
) -> pd.DataFrame:
    """
    Generates trading signals for entire DataFrame using strategy rules.
    
    Applies the evaluate_strategy function to every row in the DataFrame,
    creating a 'signal' column that indicates when to enter positions.
    This is the final step in the data preparation pipeline before backtesting
    or live trading.
    
    The function processes all historical data at once for backtesting, or
    can be used on rolling windows for live trading applications.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data and calculated technical indicators.
        Must have all features required by evaluate_strategy:
        close, ema_20, ema_50, rsi_14, atr_14, volume_ratio
    symbol : str, default "ETHUSDT"
        Trading pair symbol (used for saving processed data)
    interval : str, default "1h"
        Timeframe interval (used for saving processed data)
    is_backtest : bool, default True
        If True: saves processed data with signals to disk
        If False: skips saving (for live trading or temporary analysis)
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added 'signal' column containing:
        - 'BUY': Entry signal generated
        - 'HOLD': No action / stay in current state
    """
    df["signal"] = df.apply(evaluate_strategy, axis=1)
    
    if is_backtest:
        save_data(df, "processed", f"{symbol}_{interval}_PROCESSED.csv")
    
    return df

