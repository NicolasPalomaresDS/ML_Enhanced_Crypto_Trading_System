import pandas as pd

def rsi_oversold(row: pd.Series, threshold: int=30) -> bool:
    """
    Checks if RSI indicator is in oversold territory.
    
    Oversold conditions suggest the asset may be undervalued and due for a
    potential bounce or reversal. Traditional technical analysis considers
    RSI below 30 as oversold, indicating selling pressure may be exhausted.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated technical
        indicators. Must have 'rsi_14' column.
    threshold : int, default 30
        RSI threshold value defining oversold condition.
        Standard is 30; lower values (e.g., 20) are more conservative.
        
    Returns
    -------
    bool
        True if RSI is below threshold (oversold), False otherwise
    """
    return row["rsi_14"] < threshold

def rsi_overbought(row: pd.Series, threshold: int=70) -> bool:
    """
    Checks if RSI indicator is in overbought territory.
    
    Overbought conditions suggest the asset may be overvalued and due for a
    potential pullback or reversal. Traditional technical analysis considers
    RSI above 70 as overbought, indicating buying pressure may be exhausted.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated technical
        indicators. Must have 'rsi_14' column.
    threshold : int, default 70
        RSI threshold value defining overbought condition.
        Standard is 70; higher values (e.g., 80) are more conservative.
        
    Returns
    -------
    bool
        True if RSI is above threshold (overbought), False otherwise
    """
    return row["rsi_14"] > threshold

def ema_bullish(row: pd.Series) -> bool:
    """
    Checks if short-term EMA is above long-term EMA (bullish alignment).
    
    When EMA(20) > EMA(50), it indicates short-term momentum is stronger than
    long-term momentum, suggesting an uptrend.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated EMAs.
        Must have 'ema_20' and 'ema_50' columns.
        
    Returns
    -------
    bool
        True if EMA(20) > EMA(50) (bullish), False otherwise
    """
    return row["ema_20"] > row["ema_50"]

def ema_bearish(row: pd.Series) -> bool:
    """
    Checks if short-term EMA is below long-term EMA (bearish alignment).
    
    When EMA(20) < EMA(50), it indicates short-term momentum is weaker than
    long-term momentum, suggesting a downtrend. This helps identify when to
    avoid long positions or consider short positions.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated EMAs.
        Must have 'ema_20' and 'ema_50' columns.
        
    Returns
    -------
    bool
        True if EMA(20) < EMA(50) (bearish), False otherwise
    """
    return row["ema_20"] < row["ema_50"]


def bullish_regime(row: pd.Series) -> bool:
    """
    Confirms market is in a strong bullish regime.
    
    A bullish regime requires two conditions:
    1. Price above EMA(50): Long-term uptrend confirmation
    2. EMA(20) > EMA(50): Short-term momentum aligned with long-term trend
    
    This double confirmation reduces false signals and ensures trading with
    the dominant trend, a key principle in trend-following strategies.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated EMAs.
        Must have 'close', 'ema_20', and 'ema_50' columns.
        
    Returns
    -------
    bool
        True if both conditions are met (strong bullish regime), False otherwise
    """
    return (
        row["close"] > row["ema_50"] and
        row["ema_20"] > row["ema_50"]
    )
    
    
def sufficient_volatility(row: pd.Series, min_atr_pct: float=0.003) -> bool:
    """
    Checks if market has sufficient volatility for profitable trading.
    
    Ensures ATR represents a minimum percentage of price, filtering out
    low-volatility periods where potential profits may not justify risk
    or where spreads/fees consume most of the move.
    
    Low volatility often occurs during consolidation or in ranging markets,
    where trend-following strategies underperform.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated ATR.
        Must have 'atr_14' and 'close' columns.
    min_atr_pct : float, default 0.003
        Minimum ATR as percentage of price (0.003 = 0.3%)
        Higher values filter more aggressively for volatile conditions.
        
    Returns
    -------
    bool
        True if (ATR / price) >= min_atr_pct, False otherwise
    """
    return (row["atr_14"] / row["close"]) >= min_atr_pct


def sufficient_volume(row: pd.Series, min_ratio: float=1.0) -> bool:
    """
    Checks if current volume exceeds its moving average.
    
    Volume confirmation helps validate trend strength and avoid low-conviction
    moves. When volume is above average, it suggests genuine market interest
    and reduces the likelihood of false breakouts.
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated volume features.
        Must have 'volume_ratio' column.
    min_ratio : float, default 1.0
        Minimum volume ratio required (volume / volume_ma_20).
        1.0 = at-average, 1.5 = 50% above average, 2.0 = double average
        
    Returns
    -------
    bool
        True if volume_ratio >= min_ratio, False otherwise
    """
    return row["volume_ratio"] >= min_ratio


def rsi_pullback_bull(row: pd.Series, low: int=40, high: int=55) -> bool:
    """
    Detects RSI pullback condition within a bullish trend.
    
    Identifies temporary weakness (pullback) in an uptrend by checking if RSI
    is in a neutral zone - not oversold (which might indicate trend reversal)
    and not overbought (which would mean pullback hasn't occurred yet).
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row containing OHLCV data with pre-calculated RSI.
        Must have 'rsi_14' column.
    low : int, default 40
        Lower bound of pullback zone (below this = potentially oversold)
    high : int, default 55
        Upper bound of pullback zone (above this = no pullback yet)
        
    Returns
    -------
    bool
        True if RSI is between low and high (in pullback zone), False otherwise
    """
    return low <= row["rsi_14"] <= high

