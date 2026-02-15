from .metrics_calculator import TradeMetricsCalculator
import pandas as pd
from pathlib import Path

class TradeExtractor:
    """
    Extracts individual trade records from backtest results.
    
    This class parses a backtest DataFrame to identify and extract complete
    trade cycles (entry to exit), calculating trade-level metrics such as
    profit/loss, duration, and exit reason.
    
    Attributes
    ----------
    trade_metrics : TradeMetricsCalculator
        Calculator instance for computing trade-level metrics
    """
    def __init__(self):
        self.trade_metrics = TradeMetricsCalculator()
        
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through the backtest DataFrame to identify trade entries
        (BUY signals) and exits (rows with exit_reason), pairing them to
        create complete trade records with entry/exit details and PnL.
        
        Parameters
        ----------
        df : pd.DataFrame
            Backtest results with required columns:
            - 'signal': Trading signals (BUY/HOLD)
            - 'close': Closing prices
            - 'exit_reason': Exit reason (STOP-LOSS/TAKE-PROFIT/None)
            Index must be datetime for duration calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame where each row represents a complete trade with columns:
            - 'entry_time': Entry timestamp (index from input df)
            - 'exit_time': Exit timestamp (index from input df)
            - 'entry_price': Price at entry
            - 'exit_price': Price at exit
            - 'pnl': Profit/loss as decimal return (e.g., 0.05 = 5% gain)
            - 'duration': Time between entry and exit (Timedelta)
            - 'exit_reason': Reason for exit (STOP-LOSS/TAKE-PROFIT)
        """
        trades = []
        in_trade = False
        entry_time = None
        entry_price = None
        
        for idx, row in df.iterrows():
            signal = row["signal"]
            price = row["close"]
            
            if signal == "BUY" and not in_trade:
                in_trade = True
                entry_time = idx
                entry_price = price
                
            elif row["exit_reason"] is not None and in_trade:
                exit_time = idx
                exit_price = price
                pnl = exit_price / entry_price - 1
                
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "duration": (exit_time - entry_time),
                    "exit_reason": row["exit_reason"]
                })
                
                in_trade = False
                entry_time = None
                entry_price = None
        
        return pd.DataFrame(trades)
    
    def extract_with_metrics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Extracts trades and calculates aggregate trade metrics.
        
        Convenience method that combines trade extraction with metric calculation,
        providing both individual trade records and summary statistics in one call.
        
        Parameters
        ----------
        df : pd.DataFrame
            Backtest results with required columns (see extract() method)
            
        Returns
        -------
        tuple[pd.DataFrame, dict]
            A tuple containing:
            1. DataFrame of individual trades (same as extract() output)
            2. Dictionary of aggregate metrics:
               - 'num_trades': Total number of completed trades
               - 'trade_win_rate': Fraction of profitable trades (decimal)
               - 'avg_trade_return': Average PnL per trade (decimal)
               - 'avg_trade_duration': Average time per trade (Timedelta)
        """
        trades = self.extract(df)
        
        results = {
            "num_trades": len(trades),
            "trade_win_rate": self.trade_metrics.trade_win_rate(trades),
            "avg_trade_return": self.trade_metrics.avg_trade_return(trades),
            "avg_trade_duration": self.trade_metrics.avg_trade_duration(trades)
        }
        
        return trades, results