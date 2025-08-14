import pandas as pd
import asyncio
import os
from src.data_pipeline.exchange_collector import ExchangeDataCollector
from src.data_pipeline.indicators import TechnicalIndicatorCalculator
from src.ml_models.crypto_lstm import CryptoPredictor
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_models_for_all_symbols():
    """Train ML models for all symbols"""

    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]

    # Initialize components
    exchange_config = {"binance": {"sandbox": True}}
    data_collector = ExchangeDataCollector(exchange_config)
    indicator_calculator = TechnicalIndicatorCalculator()

    # Create models directory
    os.makedirs("models", exist_ok=True)

    for symbol in symbols:
        try:
            logger.info(f"Training model for {symbol}")

            # Get historical data (30 days)
            df = await data_collector.fetch_historical_data(symbol, "1h", 30)

            if len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue

            # Calculate technical indicators
            df_with_indicators = indicator_calculator.calculate_all_indicators(df)

            # Remove rows with NaN values
            df_clean = df_with_indicators.dropna()

            if len(df_clean) < 100:
                logger.warning(f"Insufficient clean data for {symbol}, skipping")
                continue

            # Train model
            predictor = CryptoPredictor()
            results = predictor.train(df_clean, epochs=50, batch_size=32)

            # Save model
            model_filename = f"models/{symbol.replace('/', '_')}_lstm.pth"
            predictor.save_model(model_filename)

            logger.info(
                f"Model trained for {symbol}. RMSE: {results.get('test_rmse', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")

    logger.info("Model training completed for all symbols")


if __name__ == "__main__":
    asyncio.run(train_models_for_all_symbols())
