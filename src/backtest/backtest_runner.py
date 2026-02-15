import pandas as pd
from pathlib import Path
from .backtest_engine import BacktestEngine
from .metrics_calculator import MetricsCalculator, TradeMetricsCalculator
from .trade_extractor import TradeExtractor
from .walk_forward_analyzer import WalkForwardAnalyzer
from .robustness_analyzer import RobustnessAnalyzer

class BacktestRunner:
    """
    Main orchestrator that coordinates all backtesting components.
    
    This class provides a unified interface for running various types of backtests
    and analyses. It integrates the backtest engine, metrics calculators, trade
    extractors, and specialized analyzers (walk-forward, robustness) into a single
    convenient API.
    
    Parameters
    ----------
    fee_rate : float
        Trading commission per operation as fraction (0.001 = 0.1%)
    atr_SL_mult : float
        ATR multiplier for stop-loss distance calculation
    atr_TP_mult : float
        ATR multiplier for take-profit distance calculation
    risk_pct : float
        Maximum risk per trade as fraction of equity (0.01 = 1%)
    initial_equity : float
        Starting capital for backtests
        
    Attributes
    ----------
    engine : BacktestEngine
        Core backtesting engine
    metrics_calculator : MetricsCalculator
        Calculator for equity curve metrics
    trade_extractor : TradeExtractor
        Extractor for individual trade records
    trade_metrics_calculator : TradeMetricsCalculator
        Calculator for trade-level metrics
    """
    def __init__(
        self,
        fee_rate: float,
        atr_SL_mult: float,
        atr_TP_mult: float,
        risk_pct: float,
        initial_equity: float
    ):
        self.engine = BacktestEngine(
            fee_rate=fee_rate,
            atr_SL_mult=atr_SL_mult,
            atr_TP_mult=atr_TP_mult,
            risk_pct=risk_pct,
            initial_equity=initial_equity
        )
        
        self.metrics_calculator = MetricsCalculator(initial_equity=initial_equity)
        self.trade_extractor = TradeExtractor()
        self.trade_metrics_calculator = TradeMetricsCalculator()
        
    def run_full_backtest(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Runs a complete backtest on the entire dataset.
        
        Executes the strategy across all historical data and calculates
        comprehensive performance metrics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data with:
            - DatetimeIndex
            - OHLCV columns (open, high, low, close, volume)
            - 'signal' column with trading signals
            - 'atr_14' column for position sizing
            
        Returns
        -------
        tuple[pd.DataFrame, dict]
            A tuple containing:
            1. DataFrame with backtest results including:
               - Original OHLCV data
               - 'position': Position state at each bar
               - 'equity_curve': Running equity
               - 'exit_reason': Exit reasons (STOP-LOSS/TAKE-PROFIT)
               - 'strategy_returns': Period returns
            2. Dictionary with aggregate metrics:
               - 'total_return': Overall return (decimal)
               - 'max_drawdown': Maximum drawdown (negative decimal)
               - 'win_rate': Fraction of winning trades (decimal)
        """
        df_backtest = self.engine.run(df)
        results = self.metrics_calculator.calculate_metrics(df_backtest)
        return df_backtest, results
    
    def run_oos_backtest(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.7
    ) -> tuple[pd.DataFrame, dict]:
        """
        Executes out-of-sample backtest with train/test split.
        
        Splits data into training and testing periods, then runs backtest only on
        the test (out-of-sample) period. This simulates forward testing to validate
        strategy robustness and detect overfitting.
        
        The training period is not backtested (reserved for strategy development/
        optimization in practice), while the test period simulates unseen data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete historical dataset
        train_pct : float, default 0.7
            Fraction of data to allocate to training period (0.7 = 70% train, 30% test)
            
        Returns
        -------
        tuple[pd.DataFrame, dict]
            A tuple containing:
            1. DataFrame with OOS backtest results (test period only)
            2. Dictionary with OOS performance metrics
        """
        split_idx = int(len(df) * train_pct)
        test = df.iloc[split_idx:]
        
        df_oos_backtest = self.engine.run(test)
        results = self.metrics_calculator.calculate_metrics(df_oos_backtest)
        
        return df_oos_backtest, results
    
    def extract_trades(self, df_backtest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Extracts individual trade records and calculates trade-level metrics.
        
        Parses backtest results to identify complete trade cycles (entry to exit)
        and computes statistics on individual trades such as win rate, average
        return, and average duration.
        
        Parameters
        ----------
        df_backtest : pd.DataFrame
            Backtest results from run_full_backtest() or run_oos_backtest()
            
        Returns
        -------
        tuple[pd.DataFrame, dict]
            A tuple containing:
            1. DataFrame of individual trades with columns:
               - 'entry_time': Entry timestamp
               - 'exit_time': Exit timestamp
               - 'entry_price': Entry price
               - 'exit_price': Exit price
               - 'pnl': Profit/loss as decimal return
               - 'duration': Time in trade (Timedelta)
               - 'exit_reason': STOP-LOSS or TAKE-PROFIT
            2. Dictionary with trade metrics:
               - 'num_trades': Total number of trades
               - 'trade_win_rate': Fraction of profitable trades
               - 'avg_trade_return': Mean PnL per trade
               - 'avg_trade_duration': Mean time in trades
        """
        return self.trade_extractor.extract_with_metrics(df_backtest)
    
    def calculate_expectancy(self, trades: pd.DataFrame) -> dict:
        """
        Calculates the system's expectancy metrics (mathematical edge).
        
        Computes comprehensive edge metrics including expected value per trade,
        profit factor, and win/loss statistics. Expectancy reveals whether the
        system has a positive edge and is profitable in the long run.
        
        Parameters
        ----------
        trades : pd.DataFrame
            Individual trades DataFrame from extract_trades()
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'expectancy': Expected profit per trade 
            (win_rate × avg_win + loss_rate × avg_loss)
            - 'win_rate': Fraction of winning trades (0.0 to 1.0)
            - 'avg_win': Average profit of winning trades
            - 'avg_loss': Average loss of losing trades (negative value)
            - 'profit_factor': Ratio of gross profits to gross losses 
            (>1 is profitable)
        """
        return self.trade_metrics_calculator.expectancy_metrics(trades)
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        is_days: int=90,
        oos_days: int=30
    ) -> list[dict]:
        """
        Executes walk-forward analysis for robust validation.
        
        Performs sequential train/test validation by sliding through historical
        data with fixed in-sample (training) and out-of-sample (testing) windows.
        This is the gold standard for validating strategy robustness and detecting
        overfitting.
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete historical dataset
        is_days : int, default 90
            Number of days in each in-sample (training) window
        oos_days : int, default 30
            Number of days in each out-of-sample (testing) window
            
        Returns
        -------
        list[dict]
            List of dictionaries, one per walk-forward window, each containing:
            - 'window': Window number (1-indexed)
            - 'is_total_return': In-sample total return (decimal)
            - 'is_max_drawdown': In-sample maximum drawdown (negative decimal)
            - 'oos_total_return': Out-of-sample total return (decimal)
            - 'oos_max_drawdown': Out-of-sample maximum drawdown (negative decimal)
            - 'oos_win_rate': Out-of-sample win rate (decimal)
        """
        wf_analyzer = WalkForwardAnalyzer(
            is_days=is_days,
            oos_days=oos_days,
            backtest_engine=self.engine
        )
        return wf_analyzer.run(df)
    
    def run_robustness_test(
        self,
        df: pd.DataFrame,
        sl_multipliers: list[float] = None,
        tp_multipliers: list[float] = None
    ) -> pd.DataFrame:
        """
        Executes parameter robustness test across multiple SL/TP combinations.
        
        Tests strategy performance across different stop-loss and take-profit
        settings to identify parameter sensitivity. A robust strategy should show
        consistent performance across a range of parameters, not just at one
        specific combination.
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete historical dataset
        sl_multipliers : list[float], optional
            List of ATR multipliers to test for stop-loss.
            Default: [1.2, 1.5, 1.8]
        tp_multipliers : list[float], optional
            List of ATR multipliers to test for take-profit.
            Default: [2.4, 3.0, 3.6]
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame with one row per parameter combination:
            - 'SL': Stop-loss multiplier used
            - 'TP': Take-profit multiplier used
            - 'total_return': Total return achieved (decimal)
            - 'max_drawdown': Maximum drawdown (negative decimal)
            - 'win_rate': Win rate achieved (decimal)
        """
        robustness = RobustnessAnalyzer(
            sl_multipliers=sl_multipliers,
            tp_multipliers=tp_multipliers,
            base_engine=self.engine
        )
        return robustness.run(df)