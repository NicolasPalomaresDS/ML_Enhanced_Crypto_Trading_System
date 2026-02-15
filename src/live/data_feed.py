import time
import pandas as pd
from pathlib import Path
from typing import Optional
from src.core import fetch_klines
from src.core import save_data

class DataFeed:
    """
    Provides a replay feed of historical cryptocurrency data.
    
    This class downloads historical candlestick (kline) data from the Binance API
    and replays it sequentially, simulating real-time data flow.
    
    Parameters
    ----------
    symbol : str
        Trading pair (e.g. 'BTCUSDT', 'ETHUSDT')
    interval : str
        Candlestick interval (e.g. '1h', '15m', '1d')
    start_time : str, optional
        Start date in ISO format (YYYY-MM-DD). 
        If None, fetches earliest available data
    end_time : str, optional
        End date in ISO format (YYYY-MM-DD). 
        If None, fetches up to most recent data
    sleep_seconds : float, default 0.0
        Sleep duration between candles to simulate real-time replay
        
    Attributes
    ----------
    df : pd.DataFrame
        Downloaded OHLCV data indexed by open_time
    sleep_seconds : float
        Sleep duration between candle retrievals
    pointer : int
        Current position in the data feed (starts at 0)
    """
    def __init__(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        sleep_seconds: float = 0.0
    ):
        self.df = self._download_data(
            symbol=symbol,
            interval=interval,
            dir_name="raw",
            start_time=start_time,
            end_time=end_time
        )
        
        self.sleep_seconds = sleep_seconds
        self.pointer = 0
        
    @staticmethod
    def _download_data(
        symbol: str,
        interval: str,
        dir_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Downloads historical klines (candlesticks) from Binance API
        and saves them on disk.
        
        Parameters
        ----------
        symbol : str
            Trading pair (e.g. 'BTCUSDT')
        interval : str
            Kline interval (e.g. '1h', '15m')
        dir_name : str
            Name of the directory where the data will be saved
        start_time : str, optional
            Start time in ISO format (YYYY-MM-DD)
        end_time : str, optional
            End time in ISO format (YYYY-MM-DD)
            
        Returns
        -------
        pd.DataFrame
            Normalized OHLCV time series
        """
        project_root = Path(__file__).resolve().parents[2]
        file_name = f"{symbol}_{interval}.csv"
        file_path = project_root / "data" / dir_name / file_name
        
        if file_path.exists():
            df = pd.read_csv(
                file_path, parse_dates=["open_time"],
                index_col="open_time"
            )
            return df
        
        df = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        save_data(df, dir_name, file_name)
        return df
        
    def get_latest_closed_candle(self) -> Optional[dict]:
        """
        Returns the next closed candle in replay mode.
        
        Retrieves the candle at the current pointer position and advances
        the pointer by one. Optionally sleeps to simulate real-time data flow.
        
        Returns
        -------
        dict or None
            Dictionary containing candle data with keys:
            - 'open_time': timestamp of candle opening
            - 'open': opening price
            - 'high': highest price
            - 'low': lowest price
            - 'close': closing price
            - 'volume': trading volume
            Returns None when all candles have been consumed 
            (pointer >= data length)
        """
        if self.pointer >= len(self.df):
            return None

        row = self.df.iloc[self.pointer]
        timestamp = self.df.index[self.pointer]

        candle = {
            "open_time": timestamp,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }

        self.pointer += 1

        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
            
        return candle