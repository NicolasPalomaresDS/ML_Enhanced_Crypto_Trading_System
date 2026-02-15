from pathlib import Path
import joblib
import pandas as pd

class Model:
    """
    XGBoost-based ML filter for trade signal validation.
    
    This class loads a pre-trained XGBoost classifier that acts as a gatekeeper
    for trading signals. Before executing a BUY signal, the model evaluates
    market conditions and predicts the probability of trade success. Only signals
    exceeding a confidence threshold are allowed through.
    
    The model helps reduce false positives and improve win rate by filtering out
    low-probability setups, effectively acting as a secondary confirmation layer
    on top of the base trading strategy.
    
    Attributes
    ----------
    model : xgboost.XGBClassifier
        Trained XGBoost model loaded from disk
    threshold : float
        Minimum predicted probability required to allow a trade (default: 0.6)
        Higher values = more conservative filtering, fewer but higher-quality trades
    """
    def __init__(self):
        self._load_model()
        self.threshold = 0.6
        
    def _load_model(self):
        """
        Loads the XGBClassifier model from the src/core/model directory.
        
        Locates the model file relative to the current module's location and
        deserializes the trained XGBoost classifier using joblib.
        
        The model file must exist at: src/core/model/xgb_classifier.joblib
        """
        base_path = Path(__file__).resolve().parent
        model_path = base_path / "model" / "xgb_classifier.joblib"
        self.model = joblib.load(model_path)
        
    def _model_features(self, df: pd.DataFrame, i: int) -> pd.DataFrame:
        """
        Extracts and engineers features required by the ML model.
        
        Constructs a feature vector from the specified DataFrame row, including
        both raw indicator values and derived features like EMA distance normalized
        by ATR. Features are returned in the exact format expected by the trained model.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with technical indicators already calculated.
            Must contain columns: ema_20, rsi_14, atr_14, sma_20, volume_ratio,
            volume_ma_20, close
        i : int
            Integer index position of the row to evaluate
            
        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with model features:
            - 'day_of_week': Day of week (0=Monday, 6=Sunday)
            - 'hour_of_trade': Hour of day (0-23)
            - 'ema_20': 20-period EMA value
            - 'rsi_14': 14-period RSI value
            - 'atr_14': 14-period ATR value
            - 'sma_20': 20-period SMA value
            - 'volume_ratio': Volume / Volume MA(20)
            - 'volume_ma_20': 20-period volume MA
            - 'close': Closing price
            - 'ema_distance': (close - ema_20) / atr_14 (normalized distance)
        """
        row = df.iloc[i]
        features = {
            "day_of_week": row.name.dayofweek,
            "hour_of_trade": row.name.hour,
            "ema_20": row["ema_20"],
            "rsi_14": row["rsi_14"],
            "atr_14": row["atr_14"],
            "sma_20": row["sma_20"],
            "volume_ratio": row["volume_ratio"],
            "volume_ma_20": row["volume_ma_20"],
            "close": row["close"],
            "ema_distance": (row["close"] - row["ema_20"]) / row["atr_14"]
        }
        return pd.DataFrame([features])
    
    def filter_allows(self, df: pd.DataFrame, i: int) -> bool:
        """
        Evaluates whether a trade signal should be allowed based on ML prediction.
        
        Extracts features from the specified DataFrame row, runs them through the
        trained XGBoost classifier, and compares the predicted probability of
        success against the configured threshold. Acts as a binary gate: either
        the trade passes the filter or it doesn't.
        
        This method is called before executing BUY signals to add an ML-based
        confirmation layer, filtering out low-probability setups.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with all required technical indicators calculated.
            Index must be DatetimeIndex for temporal feature extraction.
        i : int
            Integer index position of the row to evaluate (typically the most
            recent bar in live trading, or current bar in backtesting)
            
        Returns
        -------
        bool
            True if:
            - Model predicts probability >= threshold, OR
            - Model is not loaded (fail-safe: allow all trades if model unavailable)
            
            False if:
            - Model predicts probability < threshold
        """
        if self.model is None:
            return True
        
        X_row = self._model_features(df, i)
        proba = self.model.predict_proba(X_row)[0, 1]
        
        return proba >= self.threshold