import pandas as pd
from typing import Optional

class MetricsCalculator:
    """
    Calculates performance metrics for backtesting results.
    
    This class computes key performance indicators from a backtest DataFrame
    including total return, maximum drawdown, and win rate. Metrics are
    calculated based on equity curve and exit reasons.
    
    Parameters
    ----------
    initial_equity : float
        Starting capital used for return calculations
        
    Attributes
    ----------
    initial_equity : float
        Initial capital (immutable reference for calculations)
    """
    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        
    def total_return(self, df: pd.DataFrame) -> float:
        """
        Calculates the total return of the strategy.
        
        Compares final equity to initial equity to determine overall
        performance across the entire backtest period.
        
        Parameters
        ----------
        df : pd.DataFrame
            Backtest results with 'equity_curve' column
            
        Returns
        -------
        float
            Total return as decimal (e.g., 0.25 = 25% gain, -0.15 = 15% loss)
        """
        final_equity = df["equity_curve"].iloc[-1]
        return (final_equity - self.initial_equity) / self.initial_equity


    def max_drawdown(self, df: pd.DataFrame) -> float:
        """
        Calculates the maximum drawdown from peak equity.
        
        Measures the largest peak-to-trough decline in the equity curve,
        representing the worst-case loss from a historical high point.
        This is a key risk metric.
        
        Parameters
        ----------
        df : pd.DataFrame
            Backtest results with 'equity_curve' column
            
        Returns
        -------
        float
            Maximum drawdown as negative decimal (e.g., -0.2 = 20% max loss 
            from peak, -0.05 = 5% max loss from peak)
        """
        cum_max = df["equity_curve"].cummax()
        drawdown = (df["equity_curve"] - cum_max) / cum_max
        return drawdown.min()


    def win_rate(self, df: pd.DataFrame) -> float:
        """
        Calculates the percentage of winning trades.
        
        Determines what fraction of closed trades resulted in take-profit
        exits (wins) versus stop-loss exits (losses).
        
        Parameters
        ----------
        df : pd.DataFrame
            Backtest results with 'exit_reason' column containing
            'TAKE-PROFIT' or 'STOP-LOSS' values
            
        Returns
        -------
        float
            Win rate as decimal (e.g., 0.65 = 65% winning trades, 0.40 = 40% wins)
            Returns 0.0 if no trades were executed
        """
        exits = df[df["exit_reason"].isin(["TAKE-PROFIT", "STOP-LOSS"])]
        if exits.empty:
            return 0.0
        wins = exits["exit_reason"] == "TAKE-PROFIT"
        return wins.mean()
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """
        Calculates all performance metrics in a single call.
        
        Convenience method that computes total return, maximum drawdown,
        and win rate, returning them in a dictionary.
        
        Parameters
        ----------
        df : pd.DataFrame
            Backtest results with required columns:
            - 'equity_curve': equity at each bar
            - 'exit_reason': trade exit reasons
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'total_return': overall strategy return (decimal)
            - 'max_drawdown': worst drawdown from peak (negative decimal)
            - 'win_rate': fraction of winning trades (decimal)
        """
        return {
            "total_return": self.total_return(df),
            "max_drawdown": self.max_drawdown(df),
            "win_rate": self.win_rate(df)
        }
        
        
class TradeMetricsCalculator:
    """
    Calculates metrics for individual trades.
    
    This class provides static methods to analyze trade-level performance
    including win rate, average returns, trade duration, and expectancy metrics.
    Used for detailed trade analysis and system edge calculation.
    
    Methods are static as they don't require instance state.
    """
    @staticmethod
    def trade_win_rate(trades: pd.DataFrame) -> float:
        """
        Calculates win rate from individual trade records.
        
        Determines the fraction of trades that were profitable (PnL > 0).
        
        Parameters
        ----------
        trades : pd.DataFrame
            DataFrame containing individual trades with 'pnl' column
            (profit/loss for each trade)
            
        Returns
        -------
        float
            Win rate as decimal (e.g., 0.40 = 40% of trades were winners)
            Returns 0.0 if trades DataFrame is empty
        """
        if trades.empty:
            return 0.0
        return (trades["pnl"] > 0).mean()
    
    @staticmethod
    def avg_trade_return(trades: pd.DataFrame) -> float:
        """
        Calculates average profit/loss per trade.
        
        Computes the mean PnL across all trades, which represents the
        expected return per trade (system edge if positive).
        
        Parameters
        ----------
        trades : pd.DataFrame
            DataFrame containing individual trades with 'pnl' column
            
        Returns
        -------
        float
            Average PnL per trade as absolute value (e.g., 15.5 = $15.50
            avg per trade, -3.2 = average loss of $3.20)
            Returns 0.0 if trades DataFrame is empty
        """
        if trades.empty:
            return 0.0
        return trades["pnl"].mean()
    
    @staticmethod
    def avg_trade_duration(trades: pd.DataFrame) -> Optional[pd.Timedelta]:
        """
        Calculates average time spent in trades.
        
        Determines the mean duration from entry to exit across all trades,
        useful for understanding holding periods and capital efficiency.
        
        Parameters
        ----------
        trades : pd.DataFrame
            DataFrame containing individual trades with 'duration' column
            (pd.Timedelta representing time from entry to exit)
            
        Returns
        -------
        pd.Timedelta or None
            Average trade duration as Timedelta object
            Returns Timedelta(0) if trades DataFrame is empty
        """
        if trades.empty:
            return pd.Timedelta(0)
        return trades["duration"].mean()
    
    @staticmethod
    def expectancy_metrics(trades: pd.DataFrame) -> dict[str, float]:
        """
        Calculates the system's expectancy and edge metrics.
        
        Computes comprehensive metrics that define the trading system's
        mathematical edge, including expectancy (expected value per trade),
        win/loss statistics, and profit factor.
        
        Parameters
        ----------
        trades : pd.DataFrame
            DataFrame containing individual trades with 'pnl' column
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'expectancy': Expected profit per trade 
            (win_rate * avg_win + loss_rate * avg_loss)
            - 'win_rate': Fraction of winning trades (0.0 to 1.0)
            - 'avg_win': Average profit of winning trades
            - 'avg_loss': Average loss of losing trades (negative value)
            - 'profit_factor': Ratio of gross profits to gross losses 
            (>1 is profitable)
        """
        pnl = trades["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        
        win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
        loss_rate = 1 - win_rate
        
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = losses.mean() if not losses.empty else 0.0
        
        expectancy = win_rate * avg_win + loss_rate * avg_loss
        
        profit_factor = (
            wins.sum() / abs(losses.sum())
            if not losses.empty else float("inf")
        )
        
        return {
            "expectancy": expectancy,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }