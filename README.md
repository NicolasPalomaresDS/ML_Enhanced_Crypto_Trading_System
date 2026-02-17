# ML-Enhanced Crypto Trading System: Backtesting & Live Simulation Framework

The objective of this project is to design and implement a
complete algorithmic trading system, starting from historical
data analysis and ending with a replay-based execution
framework that simulates live trading behavior. The system is
not intended to place real trades automatically, but to
generate structured BUY/HOLD recommendations while
managing positions through predefined risk rules such as
stop-loss and take-profit levels. The project also addresses 
a machine learning model that works as a trade quality filter. 
Instead of attempting to predict price movements, the model learns to estimate
whether a given strategy signal is likely to result in a profitable
trade, based on historical outcomes.

## Documentation

📄 **[Full Project Report (PDF)](./ML_Enhanced_Crypto_Trading_System.pdf)**

Technical documentation covering:
- System architecture and design decisions
- Strategy methodology and rule definitions
- Backtesting results and performance analysis
- Walk-forward validation and robustness testing
- Limitations and future improvements

## Directory Structure

```
trading-bot/   
│
├── notebooks/
│   ├── model_building/
│   │   ├── BASE_BACKTEST.csv     # Backtest results for model training
│   │   ├── BASE_TRADES.csv       # Individual trades for model training
│   │   └── model_building.ipynb  # Model training & evaluation
│   └── analysis.ipynb            # Backtest results exploration & visualization
│
├── src/
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py        # Core backtesting simulation logic
│   │   ├── backtest_runner.py        # Orchestrates full backtest workflow
│   │   ├── metrics_calculator.py     # Performance metrics
│   │   ├── robustness_analyzer.py    # Parameter sensitivity testing
│   │   ├── trade_extractor.py        # Extracts individual trades from backtest
│   │   └── walk_forward_analyzer.py  # Sequential train/test validation
│   │
│   ├── core/
│   │   ├── model/
│   │   │   └── xgb_classifier.joblib  # Trained XGBoost trade filter
│   │   ├── __init__.py
│   │   ├── data_loader.py  # Binance API client & data fetching 
│   │   ├── features.py     # Technical indicator calculation
│   │   ├── model.py        # ML model wrapper & trade filtering
│   │   ├── rules.py        # Strategy condition functions
│   │   ├── strategy.py     # Signal generation logic
│   │   └── utils.py        # I/O helpers & configuration loader
│   │
│   ├── live/
│   │   ├── __init__.py
│   │   ├── data_feed.py             # Sequential candle feed (replay mode)    
│   │   ├── live_strategy_runner.py  # Live execution orchestrator 
│   │   └── trade_engine.py          # Position & risk management
│   │
│   ├── run_backtest.py  # Entry point: full backtesting suite 
│   └── run_live.py      # Entry point: live simulation
│
├── .gitignore                             # Git ignore rules
├── config.yaml                            # System configuration      
├── README.md                              # This file
├── ML_Enhanced_Crypto_Trading_System.pdf  # Project report: methodology & results
└── requirements.txt                       # Python dependencies
```

## Requirements

First, clone this repository to your local machine:

```bash
git clone https://github.com/NicolasPalomaresDS/ML_Enhanced_Crypto_Trading_System.git
cd ML_Enhanced_Crypto_Trading_System
```

Then, it is recommended to create a virtual enviroment for dependencies management:

```bash
# Crete virtual enviroment
python -m venv venv

# Activate
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
```

Finally, you can install the necessary libraries by running the `requirements.txt` 
included in the project using the following command:

```bash
pip install -r requirements.txt
```

## Usage

The system provides two execution modes that operate independently:

### 1. Full Backtesting Suite

Runs comprehensive strategy validation including out-of-sample testing, 
walk-forward analysis, and parameter robustness testing.

```bash
# From project root directory
python -m src.run_backtest
```

**What it does:**
- Downloads/loads historical data for the configured period
- Calculates technical indicators and generates signals
- Runs full-period backtest with performance metrics
- Performs 70/30 out-of-sample validation
- Executes walk-forward analysis (9 rolling windows)
- Tests strategy robustness across SL/TP parameter combinations
- Displays results and statistics

**Output:**
- Console output with detailed metrics and trade statistics
- Generated files in `data/` directory:
  - `data/raw/` - Downloaded OHLCV candles
  - `data/processed/` - Data with calculated indicators
  - `data/backtest/` - Backtest results and individual trades

**Execution time:** ~30-60 seconds (depending on data availability)

### 2. Live Trading Simulation

Simulates real-time trading by processing historical data sequentially, 
as if candles were arriving live.

```bash
# From project root directory
python -m src.run_live
```

**What it does:**
- Processes candles one-by-one in chronological order
- Updates indicators in real-time
- Applies strategy rules and ML filter to each signal
- Manages positions with dynamic stop-loss/take-profit
- Logs trade entries and exits as they occur
- Displays final performance summary

**Output:**
- Real-time console logs showing:
  - Trade entries (price, size, SL/TP levels, fees)
  - Trade exits (reason, P&L, updated equity)
  - Final statistics (return, win rate)

**Execution time:** ~5-15 seconds

### Independence Note

⚠️ **These modes are fully independent:**
- You can run `run_live.py` without ever running `run_backtest.py`
- You can run `run_backtest.py` without ever running `run_live.py`
- Each mode downloads/processes its own data as needed

## Jupyter Notebooks

### Model Building Notebook

Trains the XGBoost trade filter model. **Can be run independently** without any prior 
execution.

```bash
jupyter notebook notebooks/model_building/model_building.ipynb
```

**Requirements:** None (includes its own training data)

**What it does:**
- Loads pre-saved backtest data for training
- Engineers features for ML model
- Trains XGBoost classifier
- Evaluates performance and selects optimal threshold
- Saves trained model to `src/core/model/xgb_classifier.joblib`

### Backtest Analysis Notebook

Exploratory analysis and visualization of backtest results.

```bash
jupyter notebook notebooks/analysis.ipynb
```

**Requirements:** Must run `python -m src.run_backtest` first to generate input data.

**What it does:**
- Visualizes price, indicators, and equity curve
- Analyzes trade distribution and exit reasons
- Explores RSI entry patterns and expectancy
- Creates performance charts

⚠️ **Important:** This notebook requires backtest output files:
- `data/processed/ETHUSDT_1h_PROCESSED.csv`
- `data/backtest/ETHUSDT_1h_BT.csv`
- `data/backtest/ETHUSDT_1h_TRADES.csv`

If these files don't exist, run the backtest first:

```bash
python -m src.run_backtest
```

## Quick Start Examples

### First-time Setup and Full Exploration

```bash
# 1. Configure parameters (optional - defaults are sensible)
vim config.yaml

# 2. Run comprehensive backtest
python -m src.run_backtest

# 3. Explore results interactively
jupyter notebook notebooks/analysis.ipynb

# 4. Test live simulation
python -m src.run_live
```

### Testing Different Configurations

```bash
# 1. Edit config.yaml (change symbol, risk, or SL/TP multipliers)
vim config.yaml

# 2. Re-run backtest with new parameters
python -m src.run_backtest

# 3. Compare live simulation behavior
python -m src.run_live
```

## Expected Console Output

### Backtest Output (sample)

```
================================================================================
  BACKTEST STRATEGY RUNNER
================================================================================

📌 Configuration:
   Symbol:          ETHUSDT
   Interval:        1h
   Period:          2024-01-01 → 2024-12-31
   Initial Equity:  $10,000.00
   Risk per Trade:  1.0%
   Fee Rate:        0.10%

--------------------------------------------------------------------------------
  1. DATA UPLOAD
--------------------------------------------------------------------------------
✓ Loading from: /path/to/data/raw/ETHUSDT_1h.csv
✓ Data Uploaded: 8,761 rows

[... detailed backtest results ...]

📊 FINAL STATISTICS
================================================================================
Initial Equity:    10000.0000
Final Equity:      17008.0800
Total Return:      +70.08%
Total Trades:      70
Win Rate:          75.71%
================================================================================
```

### Live Simulation Output (sample)

```
Starting strategy for ETHUSDT (1h)
Initial Equity: 10000.0000
Risk per trade: 1.00%
Fee rate: 0.10%
--------------------------------------------------------------------------------
[2024-03-15 14:00:00] 🟢 BUY @ 3250.50
Size: 55.3846
SL = 3180.25
TP = 3390.75
Fee = $0.055385
Equity: 9999.9446

[2024-03-18 09:00:00] 🟢 TAKE-PROFIT @ $3390.75
PnL: $7.766154 (+4.31%)
Fee = $0.055385
Equity: 10007.6554
Total Return: +0.08%

[... more trades ...]

📊 FINAL STATISTICS
================================================================================
Total Return:      +70.08%
Win Rate:          75.71%
Total Trades:      70
================================================================================
```

## Data Source

Historical market data is retrieved from the **Binance API** using the public 
`/api/v3/klines` endpoint.

**Key details:**
- **Exchange:** Binance (spot market)
- **Data type:** OHLCV candlesticks (Open, High, Low, Close, Volume)
- **Default pair:** ETHUSDT (configurable via `config.yaml`)
- **Default timeframe:** 1-hour candles (configurable)
- **Access:** No API key required for historical data download
- **Rate limits:** Automatically handled with delays between requests
- **Caching:** Downloaded data is saved locally to avoid redundant API calls

The system automatically downloads and caches data on first run. Subsequent runs 
use cached data unless new date ranges are requested.

**API Documentation:** https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data