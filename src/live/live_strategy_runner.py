import time
import pandas as pd
from typing import Optional
from .data_feed import DataFeed
from .trade_engine import TradeEngine
from src.core import build_features, evaluate_strategy, Model

class LiveStrategyRunner:
    """
    Orchestrates the execution of a trading strategy in replay mode.
    
    This class integrates data feed, feature engineering, strategy evaluation,
    ML filtering, and trade execution to simulate live trading conditions using
    historical data. It processes candles sequentially and manages the complete
    trading workflow.
    
    Parameters
    ----------
    symbol : str
        Trading pair (e.g. 'BTCUSDT', 'ETHUSDT')
    interval : str
        Candlestick interval (e.g. '1h', '15m', '1d')
    fee_rate : float
        Trading fee rate (0.001 = 0.1%)
    risk_pct : float
        Percentage of equity to risk per trade (0.005 = 0.5%)
    initial_equity : float
        Starting account balance
    atr_SL_mult : float
        ATR multiplier for stop-loss distance
    atr_TP_mult : float
        ATR multiplier for take-profit distance
    start_time : str, optional
        Start date in ISO format (YYYY-MM-DD)
    end_time : str, optional
        End date in ISO format (YYYY-MM-DD)
    lookback : int, default 200
        Number of candles to keep in memory for feature calculation
        
    Attributes
    ----------
    symbol : str
        Trading pair being traded
    interval : str
        Candlestick interval
    lookback : int
        Rolling window size for historical data
    df : pd.DataFrame or None
        Current window of OHLCV data with features
    trade_engine : TradeEngine
        Handles position management and trade execution
    feed : DataFeed
        Provides sequential candle data
    model : Model
        ML model for signal filtering
    """
    def __init__(
        self,
        symbol: str,
        interval: str,
        fee_rate: float,
        risk_pct: float,
        initial_equity: float,
        atr_SL_mult: float,
        atr_TP_mult: float,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        lookback: int = 200
    ):
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.df: Optional[pd.DataFrame] = None
        
        self.trade_engine = TradeEngine(
            atr_SL_mult=atr_SL_mult,
            atr_TP_mult=atr_TP_mult,
            fee_rate=fee_rate,
            risk_pct=risk_pct,
            initial_equity=initial_equity
        )
        
        self.feed = DataFeed(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        self.model = Model()
        
    def on_new_candle(self, candle: dict):
        """
        Handles the complete trading logic for each new closed candle.
        
        Workflow:
        1. Updates the rolling window of historical data
        2. Calculates technical indicators and features
        3. Evaluates strategy signal (BUY/SELL/HOLD)
        4. Applies ML filter to BUY signals
        5. Executes trades through the trade engine
        6. Logs trade events to console
        
        Parameters
        ----------
        candle : dict
            Candle data with keys: 
            - open_time
            - open
            - high
            - low
            - close
            - volume
        """
        if self.df is None:
            self.df = pd.DataFrame([candle]).set_index("open_time")
        else:
            self.df = pd.concat(
                [self.df, pd.DataFrame([candle]).set_index("open_time")]
            )

        if len(self.df) > self.lookback:
            self.df = self.df.iloc[-self.lookback:]

        self.df = build_features(self.df, is_backtest=False)

        if self.df.empty:
            return

        # Strategy signal
        i = len(self.df) - 1
        last_row = self.df.iloc[i]
        signal = evaluate_strategy(last_row)

        # ML Filter
        if signal == "BUY":
            allowed = self.model.filter_allows(self.df, i)
            if not allowed:
                signal = "HOLD"

        self.df.loc[self.df.index[-1], "signal"] = signal

        # Market data
        price = last_row["close"]
        high = last_row["high"]
        low = last_row["low"]
        atr = last_row["atr_14"]
        timestamp = self.df.index[-1]

        # Trade engine
        event = self.trade_engine.on_signal(
            time=timestamp,
            price=price,
            atr=atr,
            signal=signal,
            high=high,
            low=low
        )

        # Logging
        if event:
            if event["type"] == "ENTRY":
                print(
                    f"[{timestamp}] 🟢 BUY @ {price:.2f}\n"
                    f"Size: {event['position_size']:.4f}\n"
                    f"SL = {event['stop_loss']:.2f}\n"
                    f"TP = {event['take_profit']:.2f}\n"
                    f"Fee = ${event['entry_fee']:.6f}\n"
                    f"Equity: {event['equity']:.4f}\n"
                )

            elif event["type"] == "EXIT":
                emoji = "🟢" if event["pnl_amount"] > 0 else "🔴"
                print(
                    f"[{timestamp}] {emoji} {event['reason']} @ ${event['price']:.2f}\n"
                    f"PnL: ${event['pnl_amount']:+.6f} ({event['pnl_pct']:+.2f}%)\n"
                    f"Fee = ${event['exit_fee']:.6f}\n"
                    f"Equity: {event['equity']:.4f}\n"
                    f"Total Return: {event['total_return']:+.2f}%\n"
                )
        # Only show HOLD if verbose mode (optional)
        # else:
        #     print(f"[{timestamp}] HOLD | Price: {price:.2f} | ATR: {atr:.2f}")

            
    def run(self):
        """
        Executes the strategy by processing all candles from the data feed.
        
        Runs a continuous loop that:
        1. Fetches the next candle from the feed
        2. Processes it through on_new_candle()
        3. Continues until all historical data is consumed
        4. Prints final performance statistics
        
        The method displays initial configuration, real-time trade logs,
        and comprehensive final statistics including returns, win rate,
        and fees paid.
        """
        print(f"\nStarting strategy for {self.symbol} ({self.interval})")
        print(f"Initial Equity: {self.trade_engine.initial_equity:.4f}")
        print(f"Risk per trade: {self.trade_engine.risk_pct*100:.2f}%")
        print(f"Fee rate: {self.trade_engine.fee_rate*100:.2f}%")
        print("-" * 80)

        while True:
            candle = self.feed.get_latest_closed_candle()

            if candle is None:
                print("-" * 80)
                print("Replay finished.")
                self._print_final_stats()
                break

            self.on_new_candle(candle)
            
    def _print_final_stats(self):
        """
        Prints comprehensive final performance statistics.
        
        Displays:
        - Initial and final equity
        - Total return percentage
        - Trade statistics (total, wins, losses, win rate)
        - Total fees paid
        - Open position status
        """
        stats = self.trade_engine.get_stats()
        print("\n📊 FINAL STATISTICS")
        print("=" * 80)
        print(f"Initial Equity:    {self.trade_engine.initial_equity:.4f}")
        print(f"Final Equity:      {stats['equity']:.4f}")
        print(f"Total Return:      {stats['total_return_pct']:+.2f}%")
        print(f"Total Trades:      {stats['num_trades']}")
        print(f"Winning Trades:    {stats['winning_trades']}")
        print(f"Losing Trades:     {stats['losing_trades']}")
        print(f"Win Rate:          {stats['win_rate']*100:.2f}%")
        print(f"Total Fees Paid:   ${stats['total_fees']:.6f}")
        print(f"Has Open Position: {stats['has_position']}")
        print("=" * 80)