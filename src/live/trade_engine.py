from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class Trade:
    """
    Represents an open trading position.
    
    This dataclass stores all essential information about an active trade,
    including entry details and risk management levels.
    
    Attributes
    ----------
    entry_time : datetime
        Timestamp when the position was opened
    entry_price : float
        Price at which the position was entered
    stop_loss : float
        Price level that triggers position closure to limit losses
    take_profit : float
        Price level that triggers position closure to secure profits
    position_size : float
        Size of the position (in base currency units)
    """
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float

class TradeEngine:
    """
    Manages trade execution, position sizing, and performance tracking.
    
    This class handles the complete trade lifecycle: opening positions based on
    signals, monitoring for stop-loss and take-profit exits, calculating fees,
    managing equity, and tracking performance statistics.
    
    Parameters
    ----------
    atr_SL_mult : float
        ATR multiplier for calculating stop-loss distance
    atr_TP_mult : float
        ATR multiplier for calculating take-profit distance
    fee_rate : float
        Trading fee per operation as fraction (0.001 = 0.1%)
    risk_pct : float
        Risk per trade as fraction of equity (0.005 = 0.5%)
    initial_equity : float
        Starting capital amount
        
    Attributes
    ----------
    position : Trade or None
        Current open position, None if no position is active
    atr_SL_mult : float
        Stop-loss ATR multiplier
    atr_TP_mult : float
        Take-profit ATR multiplier
    fee_rate : float
        Fee rate per trade
    risk_pct : float
        Percentage of equity risked per trade
    equity : float
        Current account equity (updated after each trade)
    initial_equity : float
        Starting equity (immutable reference)
    total_fees : float
        Cumulative fees paid across all trades
    num_trades : int
        Total number of completed trades
    winning_trades : int
        Number of profitable trades
    losing_trades : int
        Number of losing trades
    """
    def __init__(
        self, 
        atr_SL_mult: float, 
        atr_TP_mult: float,
        fee_rate: float,
        risk_pct: float,
        initial_equity: float
    ):
        self.position = None
        self.atr_SL_mult = atr_SL_mult
        self.atr_TP_mult = atr_TP_mult
        self.fee_rate = fee_rate
        self.risk_pct = risk_pct
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.total_fees = 0.0
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def on_signal(
        self, 
        time: datetime, 
        price: float, 
        atr: float, 
        signal: str, 
        high: float=None, 
        low: float=None
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a trading signal and manages position lifecycle.
        
        Called on each new candle close. If no position is open and signal is BUY,
        opens a new trade. If a position exists, checks for stop-loss or take-profit
        exits using the candle's high/low range.
    
        Parameters
        ----------
        time : datetime
            Candle timestamp
        price : float
            Closing price of the candle
        atr : float
            Current ATR value for position sizing and SL/TP calculation
        signal : str
            Trading signal ('BUY' or 'HOLD')
        high : float, optional
            Candle's high price for accurate take-profit detection
        low : float, optional
            Candle's low price for accurate stop-loss detection
        
        Returns
        -------
        dict or None
            Event dictionary if a trade action occurred (ENTRY or EXIT),
            None if no action was taken (HOLD signal or position monitoring)
            
            Entry event contains: 
            - type
            - time
            - price
            - stop_loss
            - take_profit
            - position_size
            - equity
            - entry_fee
            
            Exit event contains: 
            - type
            - time
            - price
            - reason
            - pnl_pct
            - pnl_amount
            - entry_price
            - position_size 
            - equity
            - exit_fee 
            - total_return
        """
        if self.position is None:
            if signal == "BUY":
                return self._open_trade(time, price, atr)
        else:
            return self._check_exit(time, price, high, low)
        
        return None
    
    def _open_trade(
        self, 
        time: datetime, 
        price: float, 
        atr: float
    ) -> Dict[str, Any]:
        """
        Opens a new position with position sizing based on risk management.
        
        Calculates stop-loss and take-profit levels using ATR multipliers,
        determines position size based on risk percentage, deducts entry fees
        from equity, and stores the position.
        
        Parameters
        ----------
        time : datetime
            Entry timestamp
        price : float
            Entry price
        atr : float
            Current ATR value
            
        Returns
        -------
        dict or None
            Entry event dictionary with position details, or None if stop_distance
            is invalid (<=0)
        """
        stop_loss = price - self.atr_SL_mult * atr
        take_profit = price + self.atr_TP_mult * atr
        stop_distance = price - stop_loss
        
        if stop_distance <= 0:
            return None
        
        risk_ammount = self.equity * self.risk_pct
        position_size = risk_ammount / stop_distance
        entry_fee = self.fee_rate * position_size
        self.equity -= entry_fee
        self.total_fees += entry_fee
        
        self.position = Trade(
            entry_time=time,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
        
        return {
            "type": "ENTRY",
            "time": time,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "equity": self.equity,
            "entry_fee": entry_fee
        }
        
    def _check_exit(
        self, 
        time: datetime, 
        price: float, 
        high: float=None, 
        low: float=None
    ) -> Optional[Dict[str, Any]]:
        """
        Checks if the current position should be closed based on SL/TP levels.
        
        Uses the candle's high and low to detect intra-candle touches of stop-loss
        or take-profit levels. Prioritizes stop-loss over take-profit (conservative
        approach assuming worst-case scenario if both are touched in same candle).
        
        Parameters
        ----------
        time : datetime
            Current candle timestamp
        price : float
            Closing price (used as fallback if high/low not provided)
        high : float, optional
            Candle's high price
        low : float, optional
            Candle's low price
            
        Returns
        -------
        dict or None
            Exit event dictionary if SL or TP was triggered, None otherwise
        """
        if self.position is None:
            return None
        
        trade = self.position
        candle_low = low if low is not None else price
        candle_high = high if high is not None else price
        
        if candle_low <= trade.stop_loss: 
            return self._close_trade(time, trade.stop_loss, "STOP-LOSS")
        elif candle_high >= trade.take_profit:
            return self._close_trade(time, trade.take_profit, "TAKE-PROFIT")
        
        return None
            
    def _close_trade(
        self, 
        time: datetime, 
        exit_price: float, 
        reason: str
    ) -> Dict[str, Any]:
        """
        Closes the current position and updates equity and statistics.
        
        Calculates profit/loss, deducts exit fees, updates equity, increments
        trade counters, and clears the position.
        
        Parameters
        ----------
        time : datetime
            Exit timestamp
        exit_price : float
            Price at which position is closed
        reason : str
            Exit reason ('STOP-LOSS' or 'TAKE-PROFIT')
            
        Returns
        -------
        dict
            Exit event dictionary containing:
            - type: 'EXIT'
            - time: exit timestamp
            - price: exit price
            - reason: exit reason
            - pnl_pct: profit/loss as percentage
            - pnl_amount: absolute profit/loss
            - entry_price: original entry price
            - position_size: size of closed position
            - equity: updated equity after exit
            - exit_fee: fee paid on exit
            - total_return: cumulative return since initial equity
        """
        trade = self.position
        
        pnl_amount = trade.position_size * (exit_price - trade.entry_price)
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
        exit_fee = self.fee_rate * trade.position_size
        
        self.equity += pnl_amount
        self.equity -= exit_fee
        self.total_fees += exit_fee
        
        self.num_trades += 1
        if pnl_amount > 0:
            self.winning_trades += 1
        else:
            self.losing_trades +=1
        
        event = {
            "type": "EXIT",
            "time": time,
            "price": exit_price,
            "reason": reason,
            "pnl_pct": pnl_pct,
            "pnl_amount": pnl_amount,
            "entry_price": trade.entry_price,
            "position_size": trade.position_size,
            "equity": self.equity,
            "exit_fee": exit_fee,
            "total_return": (
                (self.equity - self.initial_equity)
                / self.initial_equity
                * 100
            )
        }
        
        self.position = None
        return event
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Returns current performance statistics.
        
        Computes aggregate metrics from all completed trades and current equity.
        
        Returns
        -------
        dict
            Dictionary containing:
            - equity: current account equity
            - total_return_pct: percentage return from initial equity
            - num_trades: total number of completed trades
            - winning_trades: count of profitable trades
            - losing_trades: count of losing trades
            - win_rate: fraction of winning trades (0.0 to 1.0)
            - total_fees: cumulative fees paid across all trades
            - has_position: boolean indicating if position is currently open
        """
        win_rate = (
            self.winning_trades / self.num_trades
            if self.num_trades > 0
            else 0.0
        )
        
        total_return = (
            (self.equity - self.initial_equity)
            / self.initial_equity
            * 100
        )
        
        return {
            "equity": self.equity,
            "total_return_pct": total_return,
            "num_trades": self.num_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "total_fees": self.total_fees,
            "has_position": self.position is not None
        }