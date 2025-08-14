import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Import our modules
from config.database import DatabaseManager
from data_pipeline.exchange_collector import ExchangeDataCollector
from data_pipeline.indicators import TechnicalIndicatorCalculator
from ml_models.crypto_lstm import CryptoPredictor
from sentiment.sentiment_analyzer import CryptoSentimentAnalyzer
from signals.signal_generator import TradingSignalGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/alpha_sentinel.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class AlphaSentinel:
    def __init__(self):
        """Initialize Alpha Sentinel bot"""
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
        self.running = False

        # Initialize components
        self.db_manager = DatabaseManager()

        # Exchange configuration
        exchange_config = {
            "binance": {
                "api_key": os.getenv("BINANCE_API_KEY"),
                "api_secret": os.getenv("BINANCE_API_SECRET"),
                "sandbox": True,  # Use sandbox for testing
            }
        }
        self.data_collector = ExchangeDataCollector(exchange_config)

        # Other components
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.sentiment_analyzer = CryptoSentimentAnalyzer()
        self.signal_generator = TradingSignalGenerator()

        # ML Models (load if they exist)
        self.ml_models = {}
        self._load_ml_models()

        logger.info("Alpha Sentinel initialized successfully")

    def _load_ml_models(self):
        """Load ML models for each symbol"""
        for symbol in self.symbols:
            model_path = f"models/{symbol.replace('/', '_')}_lstm.pth"
            if os.path.exists(model_path):
                try:
                    predictor = CryptoPredictor()
                    predictor.load_model(model_path)
                    self.ml_models[symbol] = predictor
                    logger.info(f"Loaded ML model for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load model for {symbol}: {e}")

    async def collect_market_data(self, symbol):
        """Collect and store market data for a symbol"""
        try:
            # Get real-time data
            market_data = await self.data_collector.fetch_real_time_data(symbol)

            if market_data:
                # Store in database
                result = self.db_manager.insert_market_data(market_data)
                if result:
                    logger.info(f"Stored market data for {symbol}")
                    return market_data
                else:
                    logger.error(f"Failed to store market data for {symbol}")

        except Exception as e:
            logger.error(f"Error collecting market data for {symbol}: {e}")

        return None

    async def analyze_symbol(self, symbol):
        """Perform complete analysis for a symbol"""
        try:
            logger.info(f"Analyzing {symbol}")

            # 1. Get recent market data from database
            recent_data = self.db_manager.get_recent_data(symbol, limit=100)

            if len(recent_data) < 20:
                logger.warning(f"Insufficient data for {symbol}, collecting more...")
                # Get historical data if we don't have enough
                historical_df = await self.data_collector.fetch_historical_data(
                    symbol, "1h", 7
                )
                if not historical_df.empty:
                    # Store historical data
                    for _, row in historical_df.iterrows():
                        self.db_manager.insert_market_data(row.to_dict())
                    recent_data = self.db_manager.get_recent_data(symbol, limit=100)

            if len(recent_data) < 20:
                logger.error(f"Still insufficient data for {symbol}")
                return

            # Convert to DataFrame
            df = pd.DataFrame(recent_data)
            df = df.sort_values("timestamp")

            # 2. Calculate technical indicators
            df_with_indicators = self.indicator_calculator.calculate_all_indicators(df)
            latest_indicators = self.indicator_calculator.get_latest_indicators(
                df_with_indicators
            )

            # Store indicators in database
            indicators_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                **latest_indicators,
            }

            try:
                self.db_manager.client.table("technical_indicators").insert(
                    indicators_data
                ).execute()
            except Exception as e:
                logger.error(f"Error storing indicators: {e}")

            # 3. ML Prediction
            ml_prediction = None
            if symbol in self.ml_models:
                try:
                    predictions = self.ml_models[symbol].predict(
                        df_with_indicators, steps_ahead=1
                    )
                    if len(predictions) > 0:
                        ml_prediction = {
                            "predicted_price": float(predictions[0]),
                            "current_price": float(df["close"].iloc[-1]),
                            "confidence": 0.7,  # Default confidence
                        }

                        # Store prediction in database
                        prediction_data = {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "model_name": "lstm",
                            "prediction_value": ml_prediction["predicted_price"],
                            "confidence_score": ml_prediction["confidence"],
                        }

                        self.db_manager.insert_prediction(prediction_data)

                except Exception as e:
                    logger.error(f"ML prediction failed for {symbol}: {e}")

            # 4. Sentiment Analysis
            sentiment_data = self.sentiment_analyzer.aggregate_sentiment_data(
                symbol.split("/")[0]
            )

            # Store sentiment data
            sentiment_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "source": "aggregated",
                "sentiment_score": sentiment_data.get("aggregate_sentiment", 0),
                "confidence": sentiment_data.get("confidence", 0),
            }

            try:
                self.db_manager.client.table("sentiment_data").insert(
                    sentiment_record
                ).execute()
            except Exception as e:
                logger.error(f"Error storing sentiment: {e}")

            # 5. Generate Trading Signal
            current_price = float(df["close"].iloc[-1])

            signal = self.signal_generator.generate_trading_signal(
                symbol=symbol,
                current_price=current_price,
                technical_indicators=latest_indicators,
                ml_prediction=ml_prediction,
                sentiment_data=sentiment_data,
            )

            # Store signal in database
            signal_data = {
                "timestamp": signal["timestamp"].isoformat(),
                "symbol": signal["symbol"],
                "signal_type": signal["signal_type"],
                "confidence": signal["confidence"],
                "price_at_signal": signal["current_price"],
                "reason": "; ".join(signal["reasons"]),
            }

            try:
                self.db_manager.client.table("trading_signals").insert(
                    signal_data
                ).execute()
                logger.info(
                    f"Generated {signal['signal_type']} signal for {symbol} with confidence {signal['confidence']:.3f}"
                )
            except Exception as e:
                logger.error(f"Error storing signal: {e}")

            return signal

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    async def run_analysis_cycle(self):
        """Run one complete analysis cycle for all symbols"""
        logger.info("Starting analysis cycle")

        tasks = []
        for symbol in self.symbols:
            # Collect fresh market data first
            await self.collect_market_data(symbol)

            # Add analysis task
            task = asyncio.create_task(self.analyze_symbol(symbol))
            tasks.append(task)

        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        for i, result in enumerate(results):
            symbol = self.symbols[i]
            if isinstance(result, Exception):
                logger.error(f"Analysis failed for {symbol}: {result}")
            else:
                logger.info(f"Analysis completed for {symbol}")

        logger.info("Analysis cycle completed")

    def schedule_jobs(self):
        """Schedule recurring jobs"""
        # Run analysis every 5 minutes
        schedule.every(5).minutes.do(lambda: asyncio.run(self.run_analysis_cycle()))

        # Update performance metrics daily
        schedule.every().day.at("00:00").do(self.update_performance_metrics)

        logger.info("Jobs scheduled successfully")

    def update_performance_metrics(self):
        """Update daily performance metrics"""
        try:
            # This would typically calculate daily performance
            # For now, just log that we're updating metrics
            logger.info("Updating performance metrics")

            # You can add more sophisticated performance calculations here
            today = datetime.now().date()

            performance_data = {
                "date": today.isoformat(),
                "total_signals": 0,  # Calculate from database
                "correct_predictions": 0,  # Calculate from database
                "accuracy_rate": 0.0,  # Calculate from database
                "profit_loss": 0.0,  # If tracking P&L
            }

            # Store in database
            self.db_manager.client.table("bot_performance").insert(
                performance_data
            ).execute()

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def run(self):
        """Run the Alpha Sentinel bot"""
        logger.info("🤖 Alpha Sentinel Bot Starting...")

        try:
            # Schedule jobs
            self.schedule_jobs()

            # Run initial analysis
            asyncio.run(self.run_analysis_cycle())

            self.running = True
            logger.info("Alpha Sentinel is now running. Press Ctrl+C to stop.")

            # Main loop
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            logger.info("Stopping Alpha Sentinel...")
            self.running = False
        except Exception as e:
            logger.error(f"Critical error: {e}")
            self.running = False
        finally:
            logger.info("Alpha Sentinel stopped")


def main():
    """Main entry point"""
    bot = AlphaSentinel()
    bot.run()


if __name__ == "__main__":
    main()
