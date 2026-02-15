import pandas as pd
from datetime import timedelta
from src.core import load_data, save_data, Model
from pathlib import Path

class BacktestEngine:
    """
    Backtesting engine that simulates trading strategy execution on historical data.
    
    This class processes historical OHLCV data with trading signals and simulates
    a long-only trading strategy with ATR-based stop-loss and take-profit levels,
    position sizing based on risk management, and realistic trading fees. It applies
    an ML filter to BUY signals and tracks equity curve throughout the backtest.
    
    The engine assumes worst-case execution: if both SL and TP are touched in the
    same candle, stop-loss is triggered first (conservative approach).
    
    Parameters
    ----------
    fee_rate : float
        Trading commission per operation as fraction (0.001 = 0.1%)
    atr_SL_mult : float
        ATR multiplier for stop-loss distance calculation
    atr_TP_mult : float
        ATR multiplier for take-profit distance calculation
    risk_pct : float
        Maximum risk per trade as fraction of current equity (0.01 = 1%)
    initial_equity : float
        Starting capital for the backtest
        
    Attributes
    ----------
    fee_rate : float
        Commission rate per trade
    atr_SL_mult : float
        Stop-loss ATR multiplier
    atr_TP_mult : float
        Take-profit ATR multiplier
    risk_pct : float
        Risk percentage per trade
    initial_equity : float
        Starting capital (immutable)
    position : int
        Current position state (0 = no position, 1 = long position)
    entry_price : float
        Price at which current position was entered
    stop_loss : float
        Stop-loss level for current position
    take_profit : float
        Take-profit level for current position
    position_size : float
        Size of current position in base currency
    equity : float
        Current account equity (updated after each trade)
    model : Model
        ML model for filtering BUY signals
    """
    def __init__(
        self,
        fee_rate: float,
        atr_SL_mult: float,
        atr_TP_mult: float,
        risk_pct: float,
        initial_equity: float
    ):
        self.fee_rate = fee_rate
        self.atr_SL_mult = atr_SL_mult
        self.atr_TP_mult = atr_TP_mult
        self.risk_pct = risk_pct
        self.initial_equity = initial_equity
        self._reset_state()
        self.model = Model()
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes backtest on historical data with trading signals.
        
        Processes each candle sequentially, executing BUY signals 
        (filtered by ML model), monitoring for stop-loss and take-profit exits, 
        tracking position state, calculating fees, and building the equity curve.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical data with required columns:
            - "signal": Trading signal (BUY/HOLD)
            - "close": Closing price
            - "high": High price (for TP detection)
            - "low": Low price (for SL detection)
            - "atr_14": 14-period ATR for position sizing
            
        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional columns:
            - "position": Position state (0 or 1) at each bar
            - "returns": Individual trade returns (populated at exits)
            - "fees": Trading fees paid at each bar
            - "exit_reason": Reason for exit (STOP-LOSS/TAKE-PROFIT/None)
            - "equity_curve": Running account equity at each bar
            - "strategy_returns": Percentage change in equity at each bar
        """
        df = df.copy()
        self._reset_state()
        
        df["position"] = 0
        df["returns"] = 0.0
        df["fees"] = 0.0
        df["exit_reason"] = None
        df["equity_curve"] = 0.0
        df.iloc[0, df.columns.get_loc("equity_curve")] = self.equity
        
        for i in range(1, len(df)):
            self._process_bar(df, i)
            
        df["strategy_returns"] = df["equity_curve"].pct_change().fillna(0.0)
        return df
    
    def _process_bar(self, df: pd.DataFrame, i: int):
        """
        Processes a single bar to check for entries and exits.
        
        Logic flow:
        1. If no position and BUY signal → check ML filter → open position if allowed
        2. If position open and low hits SL → close position (prioritized)
        3. If position open and high hits TP → close position
        4. Update position state and equity curve
        
        Parameters
        ----------
        df : pd.DataFrame
            The backtest DataFrame being processed
        i : int
            Current bar index
        """
        signal = df.iloc[i]["signal"]
        price = df.iloc[i]["close"]
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]
        
        # BUY
        if signal == "BUY" and self.position == 0:
            if self.model.filter_allows(df, i):
                self._open_position(df, i, price)
            
        # STOP-LOSS
        elif self.position == 1 and low <= self.stop_loss:
            self._close_position(df, i, self.stop_loss, "STOP-LOSS")
            
        # TAKE-PROFIT
        elif self.position == 1 and high >= self.take_profit:
            self._close_position(df, i, self.take_profit, "TAKE-PROFIT")
            
        df.iloc[i, df.columns.get_loc("position")] = self.position
        df.iloc[i, df.columns.get_loc("equity_curve")] = self.equity
        
    def _open_position(self, df: pd.DataFrame, i: int, price: float):
        """
        Opens a new long position with risk-based position sizing.
        
        Calculates stop-loss and take-profit levels using ATR multipliers,
        determines position size based on risk percentage, deducts entry fee,
        and updates position state.
        
        Parameters
        ----------
        df : pd.DataFrame
            The backtest DataFrame
        i : int
            Current bar index
        price : float
            Entry price (close price of current bar)
        """
        atr = df.iloc[i]["atr_14"]
        self.entry_price = price
        self.stop_loss = self.entry_price - self.atr_SL_mult * atr
        self.take_profit = self.entry_price + self.atr_TP_mult * atr
        stop_distance = self.entry_price - self.stop_loss
        
        if stop_distance > 0:
            risk_ammount = self.equity * self.risk_pct
            self.position_size = risk_ammount / stop_distance
            self.position = 1
            self.equity -= self.fee_rate * self.position_size
            
    def _close_position(
        self, 
        df: pd.DataFrame, 
        i: int, 
        exit_price: float, 
        exit_reason: str
    ):
        """
        Closes the current position and updates equity.
        
        Calculates profit/loss, deducts exit fee, records exit reason,
        and resets position state.
        
        Parameters
        ----------
        df : pd.DataFrame
            The backtest DataFrame
        i : int
            Current bar index (exit bar)
        exit_price : float
            Price at which position is closed (SL or TP level)
        exit_reason : str
            Reason for exit ('STOP-LOSS' or 'TAKE-PROFIT')
        """
        pnl = self.position_size * (exit_price - self.entry_price)
        self.equity += pnl
        self.equity -= self.fee_rate * self.position_size
        df.at[df.index[i], "exit_reason"] = exit_reason
        self.position = 0
        self.entry_price = self.stop_loss = self.take_profit = 0.0
        self.position_size = 0.0
        
    def _reset_state(self):
        """
        Resets all position and equity state variables.
        
        Called before each backtest run to ensure clean state.
        Resets position status, price levels, position size, and
        equity to initial values.
        """
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_size = 0.0
        self.equity = self.initial_equity
        