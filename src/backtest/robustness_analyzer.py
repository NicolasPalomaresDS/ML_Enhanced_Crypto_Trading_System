import pandas as pd
from .backtest_engine import BacktestEngine
from .metrics_calculator import MetricsCalculator
from src.core import load_config

class RobustnessAnalyzer:
    """
    Analyzes strategy robustness across different parameter combinations.
    
    This class performs parameter sensitivity analysis by testing a trading strategy
    across multiple combinations of stop-loss and take-profit multipliers. It helps
    identify whether the strategy's performance is dependent on specific parameter
    values (fragile) or remains consistent across a range of settings (robust).
    
    A robust strategy should show:
    - Consistent positive returns across multiple parameter combinations
    - Gradual performance degradation (not cliff-like) as parameters vary
    - Similar win rates and drawdowns across settings
    
    Parameters
    ----------
    sl_multipliers : list[float], optional
        List of ATR multipliers to test for stop-loss placement.
        Default: [1.2, 1.5, 1.8]
    tp_multipliers : list[float], optional
        List of ATR multipliers to test for take-profit placement.
        Default: [2.4, 3.0, 3.6]
    base_engine : BacktestEngine, optional
        Base engine to inherit fee_rate, risk_pct, and initial_equity from.
        If None, uses default values (fee_rate=0.001, risk_pct=0.01, 
        initial_equity=10000.0)
        
    Attributes
    ----------
    sl_multipliers : list[float]
        Stop-loss multipliers to test
    tp_multipliers : list[float]
        Take-profit multipliers to test
    fee_rate : float
        Trading fee rate (inherited from base_engine or default)
    risk_pct : float
        Risk percentage per trade (inherited from base_engine or default)
    initial_equity : float
        Starting capital (inherited from base_engine or default)
    """
    def __init__(
        self,
        sl_multipliers: list[float] = None,
        tp_multipliers: list[float] = None,
        base_engine: BacktestEngine = None
    ):
        self.sl_multipliers = sl_multipliers or [1.2, 1.5, 1.8]
        self.tp_multipliers = tp_multipliers or [2.4, 3.0, 3.6]
        
        if base_engine:
            self.fee_rate = base_engine.fee_rate
            self.risk_pct = base_engine.risk_pct
            self.initial_equity = base_engine.initial_equity
        else:
            config = load_config()
            
            self.fee_rate = config["strategy"]["fee_rate"]
            self.risk_pct = config["strategy"]["fee_rate"]
            self.initial_equity = config["strategy"]["fee_rate"]
            
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes robustness test across all parameter combinations.
        
        Tests the strategy with every combination of SL and TP multipliers,
        running a complete backtest for each and recording performance metrics.
        This creates a parameter sensitivity map showing how performance varies
        with different risk/reward settings.
        
        Total combinations tested = len(sl_multipliers) × len(tp_multipliers)
        
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
        pd.DataFrame
            Results DataFrame with one row per parameter combination, containing:
            - 'SL': Stop-loss ATR multiplier used
            - 'TP': Take-profit ATR multiplier used
            - 'total_return': Total return achieved (decimal)
            - 'max_drawdown': Maximum drawdown experienced (negative decimal)
            - 'win_rate': Win rate achieved (decimal)
        """
        robustness_results = []
        
        for sl in self.sl_multipliers:
            for tp in self.tp_multipliers:
                engine = BacktestEngine(
                    fee_rate=self.fee_rate,
                    atr_SL_mult=sl,
                    atr_TP_mult=tp,
                    risk_pct=self.risk_pct,
                    initial_equity=self.initial_equity
                )
                
                bt = engine.run(df)
                metrics_calc = MetricsCalculator(initial_equity=self.initial_equity)
                
                robustness_results.append({
                    "SL": sl,
                    "TP": tp,
                    "total_return": metrics_calc.total_return(bt),
                    "max_drawdown": metrics_calc.max_drawdown(bt),
                    "win_rate": metrics_calc.win_rate(bt)
                })
        
        return pd.DataFrame(robustness_results)