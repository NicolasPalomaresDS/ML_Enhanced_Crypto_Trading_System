from src.live import LiveStrategyRunner
from src.core import load_config

def live_run():
    """
    Executes live trading simulation in replay mode.
    
    Runs the complete live trading workflow by simulating real-time market
    conditions using historical data. Processes candles sequentially as if
    they were arriving in real-time, applying the trading strategy, managing
    positions, and displaying trade events as they occur.
    
    This function serves as the main entry point for testing the live trading
    system in a controlled environment before deploying to actual live markets.
    It uses the exact same code path as true live trading but with historical
    data as the feed source.
    
    Configuration
    -------------
    All parameters are loaded from `config.yaml` in the project root:
    
    Data parameters (config["data"]):
    - symbol: Trading pair (e.g., 'ETHUSDT')
    - interval: Timeframe (e.g., '1h', '15m')
    - start_time: Replay start date (ISO format: 'YYYY-MM-DD')
    - end_time: Replay end date (ISO format: 'YYYY-MM-DD')
    
    Strategy parameters (config["strategy"]):
    - initial_equity: Starting capital
    - risk_pct: Risk per trade as fraction (e.g., 0.01 = 1%)
    - fee_rate: Trading fee rate (e.g., 0.001 = 0.1%)
    - atr_SL_mult: Stop-loss ATR multiplier
    - atr_TP_mult: Take-profit ATR multiplier
    
    To modify parameters, edit `config.yaml` rather than changing code.
    
    Workflow:
    1. Load configuration from config.yaml
    2. Initialize LiveStrategyRunner with config parameters
    3. Load historical data from disk or fetch from Binance API
    4. Process candles one-by-one in chronological order
    5. For each candle:
       - Update rolling window of historical data
       - Calculate technical indicators
       - Evaluate strategy signal
       - Apply ML filter to BUY signals
       - Execute trades via TradeEngine
       - Monitor open positions for SL/TP exits
       - Log trade events to console
    6. Display final performance statistics
    
    Notes
    -----
    Replay Mode vs True Live Trading:
    - Replay: Processes historical data sequentially (this function)
    - Live: Would connect to real-time WebSocket/API feeds
    - Both use identical LiveStrategyRunner logic
    
    Differences from backtesting:
    - Backtesting: Vectorized, processes all data at once
    - Live/Replay: Sequential, processes one candle at a time
    - Live has rolling window of limited history (lookback)
    - Live doesn't benefit from look-ahead bias
    """
    config = load_config()
    
    runner = LiveStrategyRunner(
        symbol=config["data"]["symbol"],
        interval=config["data"]["interval"],
        start_time=config["data"]["start_time"],
        end_time=config["data"]["end_time"],
        initial_equity=config["strategy"]["initial_equity"],
        risk_pct=config["strategy"]["risk_pct"],            
        fee_rate=config["strategy"]["fee_rate"],           
        atr_SL_mult=config["strategy"]["atr_SL_mult"],          
        atr_TP_mult=config["strategy"]["atr_TP_mult"]        
    )
    runner.run()
    
if __name__ == "__main__":
    live_run()