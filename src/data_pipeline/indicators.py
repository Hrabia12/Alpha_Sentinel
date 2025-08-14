import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List


class TechnicalIndicatorCalculator:
    def __init__(self):
        self.indicators = {}

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame"""
        if len(df) < 50:  # Need minimum data for indicators
            return df

        # Ensure we have the required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Create a copy to avoid modifying original
        df_indicators = df.copy()

        # RSI
        df_indicators["rsi"] = ta.rsi(df_indicators["close"], length=14)

        # MACD
        macd_data = ta.macd(df_indicators["close"])
        df_indicators["macd"] = macd_data["MACD_12_26_9"]
        df_indicators["macd_signal"] = macd_data["MACDs_12_26_9"]
        df_indicators["macd_histogram"] = macd_data["MACDh_12_26_9"]

        # Bollinger Bands
        bb_data = ta.bbands(df_indicators["close"], length=20)
        df_indicators["bb_upper"] = bb_data["BBU_20_2.0"]
        df_indicators["bb_middle"] = bb_data["BBM_20_2.0"]
        df_indicators["bb_lower"] = bb_data["BBL_20_2.0"]

        # Stochastic
        stoch_data = ta.stoch(
            df_indicators["high"], df_indicators["low"], df_indicators["close"]
        )
        df_indicators["stoch_k"] = stoch_data["STOCHk_14_3_3"]
        df_indicators["stoch_d"] = stoch_data["STOCHd_14_3_3"]

        # Moving Averages
        df_indicators["sma_20"] = ta.sma(df_indicators["close"], length=20)
        df_indicators["sma_50"] = ta.sma(df_indicators["close"], length=50)
        df_indicators["ema_12"] = ta.ema(df_indicators["close"], length=12)

        # Volume indicators
        df_indicators["volume_sma"] = ta.sma(df_indicators["volume"], length=20)

        # ATR (Average True Range)
        df_indicators["atr"] = ta.atr(
            df_indicators["high"], df_indicators["low"], df_indicators["close"]
        )

        return df_indicators

    def get_latest_indicators(self, df: pd.DataFrame) -> Dict:
        """Get the latest indicator values"""
        if df.empty:
            return {}

        latest = df.iloc[-1]
        return {
            "rsi": float(latest.get("rsi", 0)) if pd.notna(latest.get("rsi")) else None,
            "macd": float(latest.get("macd", 0))
            if pd.notna(latest.get("macd"))
            else None,
            "macd_signal": float(latest.get("macd_signal", 0))
            if pd.notna(latest.get("macd_signal"))
            else None,
            "bb_upper": float(latest.get("bb_upper", 0))
            if pd.notna(latest.get("bb_upper"))
            else None,
            "bb_middle": float(latest.get("bb_middle", 0))
            if pd.notna(latest.get("bb_middle"))
            else None,
            "bb_lower": float(latest.get("bb_lower", 0))
            if pd.notna(latest.get("bb_lower"))
            else None,
            "stoch_k": float(latest.get("stoch_k", 0))
            if pd.notna(latest.get("stoch_k"))
            else None,
            "stoch_d": float(latest.get("stoch_d", 0))
            if pd.notna(latest.get("stoch_d"))
            else None,
        }


# Test the indicators
def test_indicators():
    # Sample data
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")
    np.random.seed(42)

    # Generate sample OHLCV data
    close_prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 100)
    sample_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices + np.random.randn(len(dates)) * 50,
            "high": close_prices + np.abs(np.random.randn(len(dates)) * 100),
            "low": close_prices - np.abs(np.random.randn(len(dates)) * 100),
            "close": close_prices,
            "volume": np.random.randint(100, 1000, len(dates)),
        }
    )

    calculator = TechnicalIndicatorCalculator()
    df_with_indicators = calculator.calculate_all_indicators(sample_data)

    print("Sample data with indicators:")
    print(df_with_indicators[["close", "rsi", "macd", "bb_upper", "bb_lower"]].tail())

    print("\nLatest indicators:")
    latest = calculator.get_latest_indicators(df_with_indicators)
    print(latest)


if __name__ == "__main__":
    test_indicators()
