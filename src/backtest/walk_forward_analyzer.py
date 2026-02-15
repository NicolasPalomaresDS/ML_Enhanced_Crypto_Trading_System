import pandas as pd
from datetime import timedelta
from .backtest_engine import BacktestEngine
from .metrics_calculator import MetricsCalculator

class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis on trading strategy backtests.
    
    Walk-forward analysis is a validation technique that simulates realistic
    trading by sequentially splitting data into training (in-sample) and testing
    (out-of-sample) periods. The analyzer slides through historical data with
    non-overlapping test windows to evaluate strategy robustness and avoid 
    overfitting.
    
    The process:
    1. Split data into sequential IS (training) and OOS (testing) windows
    2. Run backtest on each IS window (simulates strategy development)
    3. Run backtest on corresponding OOS window (simulates live performance)
    4. Slide forward by OOS period and repeat
    
    This reveals whether strategy performance degrades in forward testing,
    indicating overfitting or non-robust parameters.
    
    Parameters
    ----------
    is_days : int, default 90
        Number of days in the in-sample (training) window
    oos_days : int, default 30
        Number of days in the out-of-sample (testing) window
    backtest_engine : BacktestEngine, optional
        Configured backtest engine instance. If None, creates 
        default BacktestEngine
        
    Attributes
    ----------
    is_days : int
        In-sample window size in days
    oos_days : int
        Out-of-sample window size in days
    backtest_engine : BacktestEngine
        Engine used for running backtests
    metrics_calculator : MetricsCalculator
        Calculator for computing performance metrics
    """
    def __init__(
        self,
        is_days: int = 90,
        oos_days: int = 30,
        backtest_engine: BacktestEngine = None
    ):
        self.is_days = is_days
        self.oos_days = oos_days
        self.backtest_engine = backtest_engine or BacktestEngine()
        
        self.metrics_calculator = MetricsCalculator(
            initial_equity=self.backtest_engine.initial_equity
        )
        
    def generate_splits(
        self, 
        df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates sequential walk-forward splits from time-series data.
        
        Creates non-overlapping windows by sliding through the data with a step
        size equal to oos_days. Each split consists of an IS window for training
        followed immediately by an OOS window for testing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Time-series DataFrame with DatetimeIndex containing OHLCV data
            and trading signals
            
        Returns
        -------
        list[tuple[pd.DataFrame, pd.DataFrame]]
            List of tuples where each tuple contains:
            - First element: In-sample DataFrame (training window)
            - Second element: Out-of-sample DataFrame (testing window)
        """
        splits = []
        start = df.index.min()
        end = df.index.max()
        
        while True:
            is_start = start
            is_end = is_start + timedelta(days=self.is_days)
            oos_end = is_end + timedelta(days=self.oos_days)
            
            if oos_end > end:
                break
            
            df_is = df.loc[is_start:is_end]
            df_oos = df.loc[is_end:oos_end]
            
            splits.append((df_is, df_oos))
            start = start + timedelta(days=self.oos_days)
            
        return splits
    
    def run(self, df: pd.DataFrame) -> list[dict[str, float]]:
        """
        Executes complete walk-forward analysis on the DataFrame.
        
        Performs the full walk-forward validation workflow:
        1. Generates all IS/OOS splits using generate_splits()
        2. For each split, runs backtest on both IS and OOS windows
        3. Calculates performance metrics for each window
        4. Returns results for all windows
        
        This reveals:
        - Whether strategy performs consistently across time periods
        - How much performance degrades from IS to OOS (overfitting indicator)
        - Robustness of strategy parameters
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete historical dataset with:
            - DatetimeIndex
            - OHLCV columns (open, high, low, close, volume)
            - 'signal' column with trading signals
            - Technical indicators required by strategy
            
        Returns
        -------
        list[dict[str, float]]
            List of dictionaries, one per walk-forward window, each containing:
            - 'window': Window number (1-indexed)
            - 'is_total_return': In-sample total return (decimal)
            - 'is_max_drawdown': In-sample maximum drawdown (negative decimal)
            - 'oos_total_return': Out-of-sample total return (decimal)
            - 'oos_max_drawdown': Out-of-sample maximum drawdown (negative decimal)
            - 'oos_win_rate': Out-of-sample win rate (decimal)
        """
        walk_forward_results = []
        splits = self.generate_splits(df)
        
        for i, (df_is, df_oos) in enumerate(splits, 1):
            bt_is = self.backtest_engine.run(df_is)
            bt_oos = self.backtest_engine.run(df_oos)
            
            walk_forward_results.append({
                "window": i,
                "is_total_return": self.metrics_calculator.total_return(bt_is),
                "is_max_drawdown": self.metrics_calculator.max_drawdown(bt_is),
                "oos_total_return": self.metrics_calculator.total_return(bt_oos),
                "oos_max_drawdown": self.metrics_calculator.max_drawdown(bt_oos),
                "oos_win_rate": self.metrics_calculator.win_rate(bt_oos)
            })
        
        return walk_forward_results
