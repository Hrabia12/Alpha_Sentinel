import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv
import pytz
from typing import Dict, List, Optional

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


def get_utc_now():
    """Get current UTC time"""
    return datetime.now(pytz.UTC)


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
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if isinstance(value, datetime):
                    # Convert datetime to ISO format string
                    if value.tzinfo is None:
                        # If naive datetime, assume UTC
                        value = pytz.UTC.localize(value)
                    cleaned[key] = value.isoformat()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    cleaned[key] = value
                elif isinstance(value, list):
                    cleaned[key] = [self._clean_data_for_json(item) for item in value]
                elif isinstance(value, dict):
                    cleaned[key] = self._clean_data_for_json(value)
                else:
                    # Convert other types to string
                    cleaned[key] = str(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_data_for_json(item) for item in data]
        else:
            return str(data)

    def _load_ml_models(self):
        """Load pre-trained ML models"""
        for symbol in self.symbols:
            model_filename = f"models/{symbol.replace('/', '_')}_lstm.pth"
            if os.path.exists(model_filename):
                try:
                    from ml_models.crypto_lstm import CryptoPredictor
                    predictor = CryptoPredictor()
                    predictor.load_model(model_filename)
                    self.ml_models[symbol] = predictor
                    logger.info(f"Loaded ML model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading ML model for {symbol}: {e}")
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

    async def collect_comprehensive_market_data(self, symbol):
        """Collect comprehensive market data to fill empty database columns"""
        try:
            # Get current market data
            current_data = await self.data_collector.fetch_real_time_data(symbol)
            
            if not current_data:
                return None
            
            # Get historical data for technical indicators
            historical_data = await self.data_collector.fetch_historical_data(symbol, limit=100)
            
            if historical_data:
                # Calculate technical indicators
                indicators = self.indicator_calculator.calculate_all_indicators(historical_data)
                
                # Get sentiment data
                sentiment_data = await self.sentiment_analyzer.analyze_sentiment(symbol)
                
                # Get market volatility metrics
                if len(historical_data) > 20:
                    recent_prices = historical_data['close'].tail(20)
                    volatility = recent_prices.pct_change().std() * 100
                    price_range = (recent_prices.max() - recent_prices.min()) / recent_prices.min() * 100
                else:
                    volatility = 0
                    price_range = 0
                
                # Enhanced market data with all available information
                enhanced_data = {
                    "symbol": symbol,
                    "timestamp": get_utc_now().isoformat(),
                    "open": current_data.get("open"),
                    "high": current_data.get("high"),
                    "low": current_data.get("low"),
                    "close": current_data.get("close"),
                    "volume": current_data.get("volume"),
                    
                    # Technical indicators
                    "rsi": indicators.get("rsi", {}).get("value"),
                    "macd": indicators.get("macd", {}).get("value"),
                    "macd_signal": indicators.get("macd", {}).get("signal"),
                    "macd_histogram": indicators.get("macd", {}).get("histogram"),
                    "bollinger_upper": indicators.get("bollinger_bands", {}).get("upper"),
                    "bollinger_middle": indicators.get("bollinger_bands", {}).get("middle"),
                    "bollinger_lower": indicators.get("bollinger_bands", {}).get("lower"),
                    "stochastic_k": indicators.get("stochastic", {}).get("k"),
                    "stochastic_d": indicators.get("stochastic", {}).get("d"),
                    "sma_20": indicators.get("sma", {}).get("20"),
                    "sma_50": indicators.get("sma", {}).get("50"),
                    "ema_12": indicators.get("ema", {}).get("12"),
                    "ema_26": indicators.get("ema", {}).get("26"),
                    
                    # Sentiment data
                    "sentiment_score": sentiment_data.get("compound_score"),
                    "sentiment_label": sentiment_data.get("label"),
                    "positive_sentiment": sentiment_data.get("positive_score"),
                    "negative_sentiment": sentiment_data.get("negative_score"),
                    "neutral_sentiment": sentiment_data.get("neutral_score"),
                    
                    # Market metrics
                    "volatility_20": volatility,
                    "price_range_20": price_range,
                    "volume_sma_20": historical_data['volume'].tail(20).mean() if len(historical_data) > 20 else None,
                    
                    # Additional market context
                    "market_trend": "bullish" if indicators.get("sma", {}).get("20", 0) > indicators.get("sma", {}).get("50", 0) else "bearish",
                    "rsi_signal": "oversold" if indicators.get("rsi", {}).get("value", 50) < 30 else "overbought" if indicators.get("rsi", {}).get("value", 50) > 70 else "neutral",
                    "macd_signal": "bullish" if indicators.get("macd", {}).get("histogram", 0) > 0 else "bearish"
                }
                
                return enhanced_data
                
        except Exception as e:
            logger.error(f"Error collecting comprehensive market data: {e}")
            return None

    async def analyze_symbol(self, symbol):
        """Analyze a symbol with comprehensive data collection and signal validation"""
        try:
            logger.info(f"Starting comprehensive analysis for {symbol}")
            
            # Collect comprehensive market data
            market_data = await self.collect_comprehensive_market_data(symbol)
            
            if not market_data:
                logger.warning(f"Could not collect market data for {symbol}")
                return
            
            # Store enhanced market data
            try:
                cleaned_data = self._clean_data_for_json(market_data)
                self.db_manager.insert_market_data(cleaned_data)
                logger.info(f"Enhanced market data stored for {symbol}")
            except Exception as e:
                logger.error(f"Error storing enhanced market data: {e}")
            
            # Get current price for validation
            current_price = market_data.get("close")
            if not current_price:
                logger.warning(f"No current price available for {symbol}")
                return
            
            # Validate all signals after 10 minutes
            await self.validate_all_signals(symbol, current_price)
            
            # Generate new trading signals
            signals = await self.signal_generator.generate_signals(symbol, market_data)
            
            # Store signals with enhanced data
            for signal in signals:
                enhanced_signal = {
                    **signal,
                    "market_conditions": {
                        "rsi": market_data.get("rsi"),
                        "macd_signal": market_data.get("macd_signal"),
                        "market_trend": market_data.get("market_trend"),
                        "volatility": market_data.get("volatility_20"),
                        "sentiment": market_data.get("sentiment_label")
                    },
                    "technical_indicators": {
                        "bollinger_position": "upper" if current_price > market_data.get("bollinger_upper", 0) else "lower" if current_price < market_data.get("bollinger_lower", 0) else "middle",
                        "sma_cross": "bullish" if market_data.get("sma_20", 0) > market_data.get("sma_50", 0) else "bearish",
                        "rsi_zone": market_data.get("rsi_signal")
                    }
                }
                
                try:
                    cleaned_signal = self._clean_data_for_json(enhanced_signal)
                    self.db_manager.insert_trading_signal(cleaned_signal)
                    logger.info(f"Enhanced trading signal stored for {symbol}")
                except Exception as e:
                    logger.error(f"Error storing enhanced trading signal: {e}")
            
            # Generate ML predictions
            predictions = await self.ml_models[symbol].predict(
                self.indicator_calculator.calculate_all_indicators(historical_data), steps_ahead=1
            )
            
            # Store predictions with enhanced context
            for prediction in predictions:
                enhanced_prediction = {
                    **prediction,
                    "market_context": {
                        "trend": market_data.get("market_trend"),
                        "volatility": market_data.get("volatility_20"),
                        "sentiment": market_data.get("sentiment_score"),
                        "rsi_zone": market_data.get("rsi_signal"),
                        "macd_signal": market_data.get("macd_signal")
                    }
                }
                
                try:
                    cleaned_prediction = self._clean_data_for_json(enhanced_prediction)
                    self.db_manager.insert_ml_prediction(cleaned_prediction)
                    logger.info(f"Enhanced ML prediction stored for {symbol}")
                except Exception as e:
                    logger.error(f"Error storing enhanced ML prediction: {e}")
            
            logger.info(f"Comprehensive analysis completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error during comprehensive analysis of {symbol}: {e}")

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

    def validate_signal_outcome(self, signal_data, current_price):
        """Validate if a previous signal was correct and calculate P&L after 10 minutes"""
        try:
            signal_type = signal_data.get("signal_type")
            signal_price = signal_data.get("price_at_signal")
            stop_loss = signal_data.get("stop_loss")
            take_profit = signal_data.get("take_profit")
            signal_timestamp = signal_data.get("timestamp")
            
            if not all([signal_type, signal_price, current_price, signal_timestamp]):
                return None
            
            signal_price = float(signal_price)
            current_price = float(current_price)
            
            # Calculate time difference since signal creation
            if isinstance(signal_timestamp, str):
                signal_time = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
            else:
                signal_time = signal_timestamp
                
            if signal_time.tzinfo is None:
                signal_time = pytz.UTC.localize(signal_time)
            
            current_utc = get_utc_now()
            time_diff_minutes = (current_utc - signal_time).total_seconds() / 60
            
            # Only validate signals that are at least 10 minutes old
            if time_diff_minutes < 10:
                return {
                    "outcome": "pending",
                    "pnl": 0.0,
                    "current_price": current_price,
                    "price_change_pct": 0.0,
                    "time_since_signal": time_diff_minutes,
                    "verification_status": "waiting_for_10min"
                }
            
            outcome = "pending"
            pnl = 0.0
            verification_status = "verified"
            
            if signal_type == "BUY":
                if current_price >= take_profit:
                    outcome = "correct"
                    pnl = (current_price - signal_price) / signal_price * 100
                elif current_price <= stop_loss:
                    outcome = "incorrect"
                    pnl = (stop_loss - signal_price) / signal_price * 100
                else:
                    # Still in progress after 10 minutes
                    pnl = (current_price - signal_price) / signal_price * 100
                    if pnl > 0:
                        outcome = "profitable"
                    else:
                        outcome = "unprofitable"
                    
            elif signal_type == "SELL":
                if current_price <= take_profit:
                    outcome = "correct"
                    pnl = (signal_price - current_price) / signal_price * 100
                elif current_price >= stop_loss:
                    outcome = "incorrect"
                    pnl = (signal_price - stop_loss) / signal_price * 100
                else:
                    # Still in progress after 10 minutes
                    pnl = (signal_price - current_price) / signal_price * 100
                    if pnl > 0:
                        outcome = "profitable"
                    else:
                        outcome = "unprofitable"
            
            return {
                "outcome": outcome,
                "pnl": pnl,
                "current_price": current_price,
                "price_change_pct": pnl,
                "time_since_signal": time_diff_minutes,
                "verification_status": verification_status
            }
            
        except Exception as e:
            logger.error(f"Error validating signal outcome: {e}")
            return None

    async def validate_all_signals(self, symbol, current_price):
        """Validate all active signals for a symbol after 10 minutes"""
        try:
            # Get all active signals for this symbol
            result = self.db_manager.client.table("trading_signals").select("*").eq("symbol", symbol).eq("status", "active").execute()
            
            if not result.data:
                return
            
            validated_count = 0
            correct_count = 0
            
            for signal in result.data:
                # Validate signal outcome
                outcome_data = self.validate_signal_outcome(signal, current_price)
                
                if outcome_data and outcome_data.get("verification_status") == "verified":
                    # Update signal with outcome
                    update_data = {
                        "outcome": outcome_data["outcome"],
                        "pnl": outcome_data["pnl"],
                        "current_price_at_verification": outcome_data["current_price"],
                        "verification_timestamp": get_utc_now().isoformat(),
                        "time_since_signal_minutes": outcome_data["time_since_signal"],
                        "verification_status": "verified"
                    }
                    
                    # If signal reached take-profit or stop-loss, mark as completed
                    if outcome_data["outcome"] in ["correct", "incorrect"]:
                        update_data["status"] = "completed"
                        update_data["completed_at"] = get_utc_now().isoformat()
                    
                    cleaned_update = self._clean_data_for_json(update_data)
                    
                    try:
                        self.db_manager.client.table("trading_signals").update(cleaned_update).eq("id", signal["id"]).execute()
                        
                        if outcome_data["outcome"] in ["correct", "profitable"]:
                            correct_count += 1
                        validated_count += 1
                        
                        logger.info(f"Signal {signal['id']} validated: {outcome_data['outcome']}, P&L: {outcome_data['pnl']:.2f}%")
                        
                    except Exception as e:
                        logger.error(f"Error updating signal outcome: {e}")
            
            # Calculate and log overall accuracy
            if validated_count > 0:
                accuracy = (correct_count / validated_count) * 100
                logger.info(f"Signal validation complete for {symbol}: {correct_count}/{validated_count} correct ({accuracy:.1f}%)")
                
        except Exception as e:
            logger.error(f"Error validating signals: {e}")

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
                if pred_time.tzinfo is None:
                    pred_time = pytz.UTC.localize(pred_time)
                
                current_utc = get_utc_now()
                if (current_utc - pred_time).total_seconds() < 300:  # 5 minutes
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
                    "completed_at": get_utc_now().isoformat()
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
                if signal_time.tzinfo is None:
                    signal_time = pytz.UTC.localize(signal_time)
                
                current_utc = get_utc_now()
                if (current_utc - signal_time).total_seconds() < 300:  # 5 minutes
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
                        "completed_at": get_utc_now().isoformat()
                    }
                    
                    cleaned_update = self._clean_data_for_json(update_data)
                    
                    try:
                        self.db_manager.client.table("trading_signals").update(cleaned_update).eq("id", signal["id"]).execute()
                        logger.info(f"Signal {signal['id']} outcome: {outcome_data['outcome']}, P&L: {outcome_data['pnl']:.2f}%")
                    except Exception as e:
                        logger.error(f"Error updating signal outcome: {e}")
                        
        except Exception as e:
            logger.error(f"Error validating previous signals: {e}")

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
