import pandas as pd
import yaml
from pathlib import Path
from typing import Optional

def load_data(dir_name: str, file_name: str) -> pd.DataFrame:
    """
    Loads OHLCV data from the project's data directory.
    
    Reads CSV files containing market data with technical indicators from
    the project's data storage structure. Automatically handles datetime
    parsing and index setting for time-series analysis.
    
    The function uses path resolution relative to the module location,
    ensuring it works correctly regardless of where the script is executed from.
    
    Parameters
    ----------
    dir_name : str
        Subdirectory name within the data folder (e.g., 'raw', 'processed')
    file_name : str
        Name of the CSV file to load (e.g., 'BTCUSDT_1h.csv')
        
    Returns
    -------
    pd.DataFrame
        Time-series DataFrame with:
        - DatetimeIndex: 'open_time' column set as index
        - OHLCV columns: open, high, low, close, volume
        - Technical indicators (if loading from 'processed' directory)
    """
    project_root = Path(__file__).resolve().parents[2]
    file_path = project_root / "data" / dir_name / file_name
    df = pd.read_csv(file_path, parse_dates=["open_time"], index_col="open_time")
    return df


def save_data(df: pd.DataFrame, dir_name: str, file_name: str) -> None:
    """
    Saves DataFrame to the project's data directory structure.
    
    Writes OHLCV data with or without technical indicators to CSV format,
    automatically creating necessary directory structure if it doesn't exist.
    Uses relative path resolution for portability across different environments.
    
    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame to save. Should have DatetimeIndex named
        'open_time' and OHLCV columns (with optional indicators).
    dir_name : str
        Subdirectory name within the data folder where file will be saved
        (e.g., 'raw', 'processed', 'backtest')
    file_name : str
        Name for the CSV file (e.g., 'BTCUSDT_1h_PROCESSED.csv')
        
    Returns
    -------
    None
        Function saves file to disk but returns nothing
    """
    project_root = Path(__file__).resolve().parents[2]
    new_dir = project_root / "data" / dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    file_path = new_dir / file_name
    df.to_csv(file_path)
    
    
def load_config(path: str | None = None) -> dict:
    """
    Loads and validates trading system configuration from YAML file.
    
    Reads configuration parameters from a YAML file and returns them as a
    nested dictionary. The function automatically locates config.yaml in the 
    project root if no path is provided..
    
    Parameters
    ----------
    path : str or None, default None
        Path to the configuration file. Can be:
        - None (default): Automatically uses `config.yaml` in project root
        - Relative path: Interpreted relative to current working directory
        - Absolute path: Used as-is
        
    Returns
    -------
    dict
        Configuration dictionary with two main sections:
        
        config["data"] - Data-related parameters:
            - symbol (str): Trading pair (e.g., 'ETHUSDT')
            - interval (str): Timeframe (e.g., '1h', '15m')
            - start_time (str): Historical data start date (ISO format)
            - end_time (str): Historical data end date (ISO format)
        
        config["strategy"] - Strategy parameters:
            - initial_equity (float): Starting capital
            - risk_pct (float): Risk per trade as fraction
            - fee_rate (float): Trading fee rate as fraction
            - atr_SL_mult (float): Stop-loss ATR multiplier
            - atr_TP_mult (float): Take-profit ATR multiplier
    """
    if path is None:
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as file:
        return yaml.safe_load(file)
