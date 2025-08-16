import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
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

    def _clean_data_for_json(self, data):
        """Clean data to make it JSON serializable"""
        import pandas as pd
        from datetime import datetime

        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if isinstance(value, (pd.Timestamp, datetime)):
                    cleaned[key] = value.isoformat()
                elif isinstance(value, (pd.Series, pd.DataFrame)):
                    # Convert pandas objects to native Python types
                    cleaned[key] = (
                        value.to_dict() if hasattr(value, "to_dict") else str(value)
                    )
                elif hasattr(value, "item"):  # numpy scalars
                    cleaned[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    cleaned[key] = [self._clean_data_for_json(item) for item in value]
                else:
                    cleaned[key] = self._clean_data_for_json(value)
            return cleaned
        elif isinstance(data, (list, tuple)):
            return [self._clean_data_for_json(item) for item in data]
        else:
            return self._clean_value(data)

    def _clean_value(self, value):
        """Clean individual values for JSON serialization"""
        import pandas as pd
        import numpy as np
        from datetime import datetime

        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, datetime):  # Handle Python datetime objects
            return value.isoformat()
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            return value.to_dict() if hasattr(value, "to_dict") else str(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif hasattr(value, "item"):  # Other numpy scalars
            return value.item()
        else:
            return value

    def _load_ml_models(self):
        """Load ML models for each symbol"""
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

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
                    # Try to provide more helpful error information
                    if "weights_only" in str(e):
                        logger.warning(f"PyTorch compatibility issue detected for {symbol}. Consider retraining the model.")
                    elif "MinMaxScaler" in str(e):
                        logger.warning(f"Sklearn compatibility issue for {symbol}. Model may need to be retrained.")
                    else:
                        logger.warning(f"Unknown error loading model for {symbol}. Check model file integrity.")
            else:
                logger.info(f"No pre-trained model found for {symbol}. Model will be trained when needed.")

    async def collect_market_data(self, symbol):
        """Collect and store market data for a symbol"""
        try:
            # Get real-time data
            market_data = await self.data_collector.fetch_real_time_data(symbol)

            if market_data:
                # Convert any Timestamp objects to strings
                cleaned_data = self._clean_data_for_json(market_data)
                
                # Validate cleaned data
                if not cleaned_data or not isinstance(cleaned_data, dict):
                    logger.error(f"Invalid cleaned data format for {symbol}: {type(cleaned_data)}")
                    return None

                # Store in database
                result = self.db_manager.insert_market_data(cleaned_data)
                if result:
                    logger.info(f"Stored market data for {symbol}")
                    return market_data
                else:
                    logger.error(f"Failed to store market data for {symbol}")
                    # Log the data that failed to store for debugging
                    logger.debug(f"Failed data for {symbol}: {cleaned_data}")
                    # Try to get more details about why it failed
                    if hasattr(self.db_manager, 'last_error'):
                        logger.error(f"Database error details: {self.db_manager.last_error}")

        except Exception as e:
            logger.error(f"Error collecting market data for {symbol}: {e}")
            # Add more context for debugging
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")

        return None

    def validate_signal_outcome(self, signal_data, current_price):
        """Validate if a previous signal was correct and calculate P&L"""
        try:
            signal_type = signal_data.get("signal_type")
            signal_price = signal_data.get("price_at_signal")
            stop_loss = signal_data.get("stop_loss")
            take_profit = signal_data.get("take_profit")
            
            if not all([signal_type, signal_price, current_price]):
                return None
            
            signal_price = float(signal_price)
            current_price = float(current_price)
            
            outcome = "pending"
            pnl = 0.0
            
            if signal_type == "BUY":
                if current_price >= take_profit:
                    outcome = "correct"
                    pnl = (current_price - signal_price) / signal_price * 100
                elif current_price <= stop_loss:
                    outcome = "incorrect"
                    pnl = (stop_loss - signal_price) / signal_price * 100
                else:
                    # Still in progress
                    pnl = (current_price - signal_price) / signal_price * 100
                    
            elif signal_type == "SELL":
                if current_price <= take_profit:
                    outcome = "correct"
                    pnl = (signal_price - current_price) / signal_price * 100
                elif current_price >= stop_loss:
                    outcome = "incorrect"
                    pnl = (signal_price - stop_loss) / signal_price * 100
                else:
                    # Still in progress
                    pnl = (signal_price - current_price) / signal_price * 100
            
            return {
                "outcome": outcome,
                "pnl": pnl,
                "current_price": current_price,
                "price_change_pct": pnl
            }
            
        except Exception as e:
            logger.error(f"Error validating signal outcome: {e}")
            return None

    async def validate_ml_predictions(self, symbol, current_price):
        """Validate outcomes of previous ML predictions for a symbol"""
        try:
            # Get active predictions for this symbol
            result = self.db_manager.client.table("ml_predictions").select("*").eq("symbol", symbol).eq("status", "active").execute()
            
            if not result.data:
                return
            
            for prediction in result.data:
                # Skip predictions that are too recent (less than 5 minutes old)
                pred_time = datetime.fromisoformat(prediction["timestamp"].replace('Z', '+00:00'))
                if (datetime.now() - pred_time).total_seconds() < 300:  # 5 minutes
                    continue
                
                predicted_price = float(prediction["prediction_value"])
                actual_price = float(prediction["actual_value"])
                
                # Calculate prediction accuracy (within 5% threshold)
                price_error = abs(predicted_price - actual_price) / actual_price
                is_correct = price_error <= 0.05  # 5% threshold
                
                # Update prediction with outcome
                update_data = {
                    "actual_value": current_price,  # Update with latest price
                    "prediction_accuracy": 1.0 - price_error,
                    "is_correct": is_correct,
                    "status": "completed",
                    "completed_at": datetime.now().isoformat()
                }
                
                cleaned_update = self._clean_data_for_json(update_data)
                
                try:
                    self.db_manager.client.table("ml_predictions").update(cleaned_update).eq("id", prediction["id"]).execute()
                    logger.info(f"ML prediction {prediction['id']} validated: {'correct' if is_correct else 'incorrect'}, error: {price_error:.2%}")
                except Exception as e:
                    logger.error(f"Error updating ML prediction outcome: {e}")
                    
        except Exception as e:
            logger.error(f"Error validating ML predictions: {e}")

    async def validate_previous_signals(self, symbol, current_price):
        """Validate outcomes of previous active signals for a symbol"""
        try:
            # Get active signals for this symbol
            result = self.db_manager.client.table("trading_signals").select("*").eq("symbol", symbol).eq("status", "active").execute()
            
            if not result.data:
                return
            
            for signal in result.data:
                # Skip signals that are too recent (less than 5 minutes old)
                signal_time = datetime.fromisoformat(signal["timestamp"].replace('Z', '+00:00'))
                if (datetime.now() - signal_time).total_seconds() < 300:  # 5 minutes
                    continue
                
                # Validate the signal outcome
                outcome_data = self.validate_signal_outcome(signal, current_price)
                
                if outcome_data and outcome_data["outcome"] != "pending":
                    # Update signal with outcome
                    update_data = {
                        "outcome": outcome_data["outcome"],
                        "pnl": outcome_data["pnl"],
                        "current_price": outcome_data["current_price"],
                        "status": "completed",
                        "completed_at": datetime.now().isoformat()
                    }
                    
                    cleaned_update = self._clean_data_for_json(update_data)
                    
                    try:
                        self.db_manager.client.table("trading_signals").update(cleaned_update).eq("id", signal["id"]).execute()
                        logger.info(f"Signal {signal['id']} outcome: {outcome_data['outcome']}, P&L: {outcome_data['pnl']:.2f}%")
                    except Exception as e:
                        logger.error(f"Error updating signal outcome: {e}")
                        
        except Exception as e:
            logger.error(f"Error validating previous signals: {e}")

    async def analyze_symbol(self, symbol):
        """Perform complete analysis for a symbol"""
        try:
            logger.info(f"Analyzing {symbol}")

            # Import pandas here to avoid potential issues
            import pandas as pd

            # 1. Get recent market data from database
            recent_data = self.db_manager.get_recent_data(symbol, limit=100)

            if len(recent_data) < 20:
                logger.warning(f"Insufficient data for {symbol}, collecting more...")
                # Get historical data if we don't have enough
                historical_df = await self.data_collector.fetch_historical_data(
                    symbol, "1h", 7
                )
                if not historical_df.empty:
                    # Store historical data with proper data cleaning
                    for _, row in historical_df.iterrows():
                        row_dict = row.to_dict()
                        cleaned_row = self._clean_data_for_json(row_dict)
                        self.db_manager.insert_market_data(cleaned_row)
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

            # Clean indicators data for JSON serialization
            cleaned_indicators = self._clean_data_for_json(indicators_data)

            try:
                self.db_manager.client.table("technical_indicators").insert(
                    cleaned_indicators
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
                        current_price = float(df["close"].iloc[-1])
                        predicted_price = float(predictions[0])
                        
                        ml_prediction = {
                            "predicted_price": predicted_price,
                            "current_price": current_price,
                            "confidence": 0.7,  # Default confidence
                        }

                        # Store prediction in database with more details
                        prediction_data = {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "model_name": "lstm",
                            "prediction_value": predicted_price,
                            "actual_value": current_price,  # Current price becomes actual value
                            "confidence_score": ml_prediction["confidence"],
                            "prediction_type": "price_prediction",
                            "timeframe": "1_step_ahead",
                            "status": "active"
                        }

                        cleaned_prediction = self._clean_data_for_json(prediction_data)
                        self.db_manager.insert_prediction(cleaned_prediction)
                        
                        logger.info(f"ML prediction recorded for {symbol}: {predicted_price:.2f} (current: {current_price:.2f})")

                except Exception as e:
                    logger.error(f"ML prediction failed for {symbol}: {e}")
            else:
                logger.info(f"No ML model available for {symbol}")

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

            cleaned_sentiment = self._clean_data_for_json(sentiment_record)

            try:
                self.db_manager.client.table("sentiment_data").insert(
                    cleaned_sentiment
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
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "reason": "; ".join(signal["reasons"]),
                "status": "active",  # Track signal status
                "technical_reasons": [r for r in signal["reasons"] if any(tech in r.lower() for tech in ["rsi", "macd", "bollinger", "stochastic", "moving average"])],
                "sentiment_reasons": [r for r in signal["reasons"] if any(sent in r.lower() for sent in ["sentiment", "fear", "greed", "news"])],
                "ml_reasons": [r for r in signal["reasons"] if any(ml in r.lower() for ml in ["prediction", "ml", "model"])]
            }

            cleaned_signal = self._clean_data_for_json(signal_data)

            try:
                result = self.db_manager.client.table("trading_signals").insert(
                    cleaned_signal
                ).execute()
                logger.info(
                    f"Generated {signal['signal_type']} signal for {symbol} with confidence {signal['confidence']:.3f}"
                )
                
                # Now validate previous signals and ML predictions for this symbol
                await self.validate_previous_signals(symbol, current_price)
                await self.validate_ml_predictions(symbol, current_price)
                
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
            logger.info("Updating performance metrics")
            
            # Get today's signals
            today = datetime.now().date()
            today_start = datetime.combine(today, datetime.min.time())
            today_end = datetime.combine(today, datetime.max.time())
            
            # Get signals from today
            signals_result = self.db_manager.client.table("trading_signals").select("*").gte("timestamp", today_start.isoformat()).lte("timestamp", today_end.isoformat()).execute()
            
            total_signals = len(signals_result.data) if signals_result.data else 0
            correct_predictions = 0
            total_pnl = 0.0
            
            if signals_result.data:
                for signal in signals_result.data:
                    # Check if signal outcome is recorded
                    if signal.get("outcome") == "correct":
                        correct_predictions += 1
                    if signal.get("pnl"):
                        total_pnl += float(signal.get("pnl", 0))
            
            accuracy_rate = correct_predictions / total_signals if total_signals > 0 else 0.0
            
            performance_data = {
                "date": today.isoformat(),
                "total_signals": total_signals,
                "correct_predictions": correct_predictions,
                "accuracy_rate": accuracy_rate,
                "profit_loss": total_pnl,
                "timestamp": datetime.now().isoformat()
            }

            cleaned_performance = self._clean_data_for_json(performance_data)

            # Store in database
            self.db_manager.client.table("bot_performance").insert(
                cleaned_performance
            ).execute()
            
            logger.info(f"Performance metrics updated: {total_signals} signals, {correct_predictions} correct, {accuracy_rate:.2%} accuracy")

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
