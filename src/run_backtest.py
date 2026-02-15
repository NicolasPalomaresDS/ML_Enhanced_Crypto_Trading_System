from src.core import (
    fetch_klines,
    build_features, 
    generate_signals, 
    save_data, 
    validate_time_series,
    load_data,
    load_config
)
from src.backtest import BacktestRunner
import pandas as pd
from pathlib import Path


def print_header(title: str):
    """
    Prints a formatted header for major sections.
    
    Creates a visually distinct header with double-line borders to separate
    major sections of the backtest output.
    
    Parameters
    ----------
    title : str
        Header text to display
    """
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """
    Prints a formatted section separator.
    
    Creates a visually distinct section divider with single-line borders to
    organize subsections within the backtest output.
    
    Parameters
    ----------
    title : str
        Section text to display
    """
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def print_metrics(metrics: dict, title: str="Metrics"):
    """
    Prints formatted metrics dictionary with appropriate value formatting.
    
    Displays metrics in a clean, aligned format with automatic formatting based
    on metric type and naming conventions. Percentages, decimals, and timedeltas
    are formatted appropriately for readability.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric names (keys) and values (values)
        Values can be float, int, pd.Timedelta, or other types
    title : str, default "Metrics"
        Section title displayed above the metrics
    """
    print(f"\n📊 {title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            
            if (
                'return' in key.lower() 
                or 'drawdown' in key.lower() 
                or 'rate' in key.lower()
            ):
                print(f"   {key:.<30} {value:>10.2%}")              
            else:
                print(f"   {key:.<30} {value:>10.4f}")
                
        elif isinstance(value, pd.Timedelta):
            print(f"   {key:.<30} {str(value):>10}")
        else:
            print(f"   {key:.<30} {value:>10}")


def load_or_fetch_data(
    symbol: str, 
    interval: str, 
    start_time: str, 
    end_time: str
) -> pd.DataFrame:
    """
    Loads data from disk if available, otherwise fetches from Binance API.
    
    Implements a caching strategy to avoid redundant API calls. Checks for
    existing CSV file first; if found, loads from disk. If not found, downloads
    from Binance, saves to disk for future use, then returns the data.
    
    Parameters
    ----------
    symbol : str
        Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
    interval : str
        Candlestick interval (e.g., '1h', '15m', '1d')
    start_time : str
        Start date in ISO format (YYYY-MM-DD)
    end_time : str
        End date in ISO format (YYYY-MM-DD)
        
    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex, validated for integrity
    """
    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = project_root / "data" / "raw"
    file_name = f"{symbol}_{interval}.csv"
    file_path = raw_data_dir / file_name
    
    if file_path.exists():
        print(f"✓ Loading from: {file_path}")
        
        df = pd.read_csv(
            file_path, 
            parse_dates=["open_time"], 
            index_col="open_time"
        )
        
    else:
        print(f"⬇ Downloading data -> {symbol}...")
        df = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        save_data(df, "raw", file_name)
        print(f"✓ Data saved in: {file_path}")
    
    validate_time_series(df)
    return df


def main():
    """
    Executes complete backtesting workflow with comprehensive analysis.
    
    This is the main entry point for the backtesting system. It orchestrates
    the entire workflow from data acquisition through final performance reporting,
    including multiple validation approaches (OOS, walk-forward, robustness).
    
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
    2. Data Upload: Load/fetch market data
    3. Feature Engineering: Calculate technical indicators
    4. Signal Generation: Apply trading strategy rules
    5. Full Backtest: Run strategy on complete dataset
    6. Trade Analysis: Extract and analyze individual trades
    7. OOS Validation: Test on held-out data (70/30 split)
    8. Walk-Forward Analysis: Sequential train/test windows
    9. Robustness Test: Parameter sensitivity analysis
    10. Final Summary: Comprehensive results report
    
    Configuration Parameters (modify in code):
    - symbol: Trading pair (default: "ETHUSDT")
    - interval: Timeframe (default: "1h")
    - start_time: Backtest start date (default: "2024-01-01")
    - end_time: Backtest end date (default: "2024-12-31")
    - initial_equity: Starting capital (default: 10000.0)
    - fee_rate: Trading fees (default: 0.001 = 0.1%)
    - atr_SL_mult: Stop-loss ATR multiplier (default: 1.8)
    - atr_TP_mult: Take-profit ATR multiplier (default: 2.4)
    - risk_pct: Risk per trade (default: 0.01 = 1%)
    
    Output Files Generated:
    - data/raw/{symbol}_{interval}.csv
      → Raw OHLCV data from API
    - data/processed/{symbol}_{interval}_PROCESSED.csv
      → Data with indicators and signals
    - data/backtest/{symbol}_{interval}_BT.csv
      → Full backtest results with equity curve
    - data/backtest/{symbol}_{interval}_TRADES.csv
      → Individual trade records
    
    Console Output Sections:
    1. Configuration display
    2. Data upload status
    3. Feature/signal generation progress
    4. Full backtest performance metrics
    5. Individual trade statistics
    6. Edge/expectancy analysis
    7. Out-of-sample validation results
    8. Walk-forward analysis statistics
    9. Robustness test parameter matrix
    10. Final summary with key metrics
    """
    
    # ================================================================
    # CONFIGURATION
    # ================================================================
    config = load_config()
    
    symbol = config["data"]["symbol"]
    interval = config["data"]["interval"]
    start_time = config["data"]["start_time"]
    end_time = config["data"]["end_time"]
    initial_equity = config["strategy"]["initial_equity"]
    risk_pct = config["strategy"]["risk_pct"]        
    fee_rate = config["strategy"]["fee_rate"]        
    atr_SL_mult = config["strategy"]["atr_SL_mult"]       
    atr_TP_mult = config["strategy"]["atr_TP_mult"] 
    
    print_header("BACKTEST STRATEGY RUNNER")
    print(f"\n📌 Configuration:")
    print(f"   Symbol:          {symbol}")
    print(f"   Interval:        {interval}")
    print(f"   Period:          {start_time} → {end_time}")
    print(f"   Initial Equity:  ${initial_equity:,.2f}")
    print(f"   Risk per Trade:  {risk_pct:.1%}")
    print(f"   Fee Rate:        {fee_rate:.2%}")
    print(f"   Stop-Loss:       {atr_SL_mult}x ATR")
    print(f"   Take-Profit:     {atr_TP_mult}x ATR")
    
    # ================================================================
    # DATA UPLOAD
    # ================================================================
    print_section("1. DATA UPLOAD")
    df = load_or_fetch_data(symbol, interval, start_time, end_time)
    print(f"✓ Data Uploaded: {len(df):,} rows")
    print(f"\nFrom: {df.index[0]}")
    print(f"To: {df.index[-1]}")
    
    # ================================================================
    # FEATURES AND SIGNALS
    # ================================================================
    print_section("2. FEATURES AND SIGNALS")
    
    print("⚙️  Generating features...")
    df = build_features(df)
    print("✓ Features generated")
    
    print("📡 Generating signals...")
    df = generate_signals(df)
    print("✓ Signals generated")
    
    save_data(df, "processed", f"{symbol}_{interval}_PROCESSED.csv")
    print("✓ Processed data saved")
    
    # ================================================================
    # BACKTEST RUNNER
    # ================================================================
    runner = BacktestRunner(
        fee_rate=fee_rate,
        atr_SL_mult=atr_SL_mult,
        atr_TP_mult=atr_TP_mult,
        risk_pct=risk_pct,
        initial_equity=initial_equity
    )
    
    # ================================================================
    # MAIN BACKTEST
    # ================================================================
    print_section("3. FULL BACKTEST")
    
    df_backtest, bt_results = runner.run_full_backtest(df)
    save_data(df_backtest, "backtest", f"{symbol}_{interval}_BT.csv")
    
    print_metrics(bt_results, "Performance")
    
    # Final equity
    final_equity = initial_equity * (1 + bt_results['total_return'])
    pnl = final_equity - initial_equity
    print(f"\n💰 Results:")
    print(f"   Initial Equity:  ${initial_equity:>12,.2f}")
    print(f"   Final Equity:    ${final_equity:>12,.2f}")
    print(f"   P&L:             ${pnl:>12,.2f}")
    
    # ================================================================
    # TRADES ANALYSIS
    # ================================================================
    print_section("4. TRADES ANALYSIS")
    
    trades, trade_results = runner.extract_trades(df_backtest)
    save_data(trades, "backtest", f"{symbol}_{interval}_TRADES.csv")
    
    print_metrics(trade_results, "Trades")
    
    # Edge metrics
    edge_metrics = runner.calculate_expectancy(trades)
    print_metrics(edge_metrics, "Edge (Expectancy)")
    
    # ================================================================
    # OUT-OF-SAMPLE BACKTEST
    # ================================================================
    print_section("5. OUT-OF-SAMPLE (70/30)")
    
    df_oos, oos_results = runner.run_oos_backtest(df=df, train_pct=0.7)
    
    print_metrics(oos_results, "OOS Performance")
    
    # OOS Trades
    oos_trades, oos_trade_results = runner.extract_trades(df_oos)
    print_metrics(oos_trade_results, "OOS Trades")
    
    # OOS Edge
    oos_edge = runner.calculate_expectancy(oos_trades)
    print_metrics(oos_edge, "OOS Edge")
    
    # ================================================================
    # WALK FORWARD ANALYSIS
    # ================================================================
    print_section("6. WALK FORWARD ANALYSIS")
    
    wf_results = runner.run_walk_forward(
        df_backtest, 
        is_days=90, 
        oos_days=30
    )
    df_wf = pd.DataFrame(wf_results)
    
    if df_wf.empty:
        print("⚠️  Not enough data for Walk Forward")
    else:
        print(f"\n📊 Analyzed windows: {len(df_wf)}")
        
        columns = [
            "is_total_return", 
            "oos_total_return", 
            "oos_max_drawdown", 
            "oos_win_rate"
        ]
        stats = df_wf[columns].describe()
        
        print("\n   Statistics OOS:")
        print(f"   {'Metric':<25} {'Mean':>12} {'Min':>12} {'Max':>12}")
        print("   " + "-" * 61)
        
        for col in columns:
            mean_val = stats.loc['mean', col]
            min_val = stats.loc['min', col]
            max_val = stats.loc['max', col]
            
            col_name = (
                col.replace('oos_', '')
                .replace('is_', '')
                .replace('_', ' ')
                .title()
            )
            print(
                f"{col_name:<25}", 
                f"{mean_val:>11.2%}", 
                f"{min_val:>11.2%}", 
                f"{max_val:>11.2%}"
            )
    
    # ================================================================
    # ROBUSTNESS TEST
    # ================================================================
    print_section("7. ROBUSTNESS TEST")
    
    rob_df = runner.run_robustness_test(df_backtest)
    
    print("\n🔬 Combinations of SL/TP tested:\n")
    
    rob_display = rob_df.copy()
    
    rob_display['total_return'] = (
        rob_display['total_return'].apply(lambda x: f"{x:>8.2%}")
    )
    
    rob_display['max_drawdown'] = (
        rob_display['max_drawdown'].apply(lambda x: f"{x:>8.2%}")
    )
    
    rob_display['win_rate'] = (
        rob_display['win_rate'].apply(lambda x: f"{x:>8.2%}")
    )
    
    print(rob_display.to_string(index=False))
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print_header("FINAL SUMMARY")
    
    print(f"\n✅ Backtest completed successfully")
    print(f"\n📈 Main Results:")
    print(f"   Total Return:        {bt_results['total_return']:>10.2%}")
    print(f"   Max Drawdown:        {bt_results['max_drawdown']:>10.2%}")
    print(f"   Win Rate:            {bt_results['win_rate']:>10.2%}")
    print(f"   Total Trades:        {len(trades):>10,}")
    print(f"   Profit Factor:       {edge_metrics['profit_factor']:>10.2f}")
    
    print(f"\n📊 OOS Validation:")
    print(f"   OOS Return:           {oos_results['total_return']:>10.2%}")
    print(f"   OOS Max Drawdown:     {oos_results['max_drawdown']:>10.2%}")
    print(f"   OOS Win Rate:         {oos_results['win_rate']:>10.2%}")
    
    print("\n" + "=" * 80)
    print()

if __name__ == "__main__":
    main()