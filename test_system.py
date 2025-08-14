import asyncio
import sys
import os

# Add src to path
sys.path.append("src")

from config.database import DatabaseManager
from data_pipeline.exchange_collector import ExchangeDataCollector
from data_pipeline.indicators import TechnicalIndicatorCalculator
from sentiment.sentiment_analyzer import CryptoSentimentAnalyzer
from signals.signal_generator import TradingSignalGenerator


async def test_all_components():
    """Test all system components"""

    print("🧪 Testing Alpha Sentinel Components")
    print("=" * 50)

    # Test 1: Database Connection
    print("1. Testing Database Connection...")
    try:
        db_manager = DatabaseManager()
        # Test a simple query
        result = db_manager.client.table("market_data").select("*").limit(1).execute()
        print("   ✅ Database connection successful")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")

    # Test 2: Exchange Data Collection
    print("\n2. Testing Exchange Data Collection...")
    try:
        exchange_config = {"binance": {"sandbox": True}}
        collector = ExchangeDataCollector(exchange_config)

        # Test real-time data
        data = await collector.fetch_real_time_data("BTC/USDT")
        if data:
            print("   ✅ Real-time data collection successful")
            print(f"   📊 BTC/USDT Price: ${data['close']:,.2f}")
        else:
            print("   ❌ Real-time data collection failed")
    except Exception as e:
        print(f"   ❌ Exchange data collection error: {e}")

    # Test 3: Technical Indicators
    print("\n3. Testing Technical Indicators...")
    try:
        # Get some historical data for testing
        historical_data = await collector.fetch_historical_data("BTC/USDT", "1h", 2)

        if not historical_data.empty and len(historical_data) > 20:
            indicator_calculator = TechnicalIndicatorCalculator()
            df_with_indicators = indicator_calculator.calculate_all_indicators(
                historical_data
            )
            latest_indicators = indicator_calculator.get_latest_indicators(
                df_with_indicators
            )

            print("   ✅ Technical indicators calculated successfully")
            print(f"   📈 RSI: {latest_indicators.get('rsi', 'N/A')}")
            print(f"   📊 MACD: {latest_indicators.get('macd', 'N/A')}")
        else:
            print("   ⚠️  Insufficient data for technical indicators")
    except Exception as e:
        print(f"   ❌ Technical indicators error: {e}")

    # Test 4: Sentiment Analysis
    print("\n4. Testing Sentiment Analysis...")
    try:
        sentiment_analyzer = CryptoSentimentAnalyzer()

        # Test Fear & Greed Index
        fg_data = sentiment_analyzer.get_fear_greed_index()
        print("   ✅ Fear & Greed Index retrieved successfully")
        print(
            f"   😨 Fear & Greed: {fg_data['fear_greed_value']} ({fg_data['fear_greed_text']})"
        )

        # Test aggregated sentiment
        aggregate = sentiment_analyzer.aggregate_sentiment_data("BTC")
        print("   ✅ Sentiment aggregation successful")
        print(f"   🎭 Aggregate Sentiment: {aggregate['aggregate_sentiment']:.3f}")

    except Exception as e:
        print(f"   ❌ Sentiment analysis error: {e}")

    # Test 5: Signal Generation
    print("\n5. Testing Signal Generation...")
    try:
        signal_generator = TradingSignalGenerator()

        # Create sample data for testing
        sample_indicators = {
            "close": 45000,
            "rsi": 45,
            "macd": 100,
            "macd_signal": 90,
            "bb_upper": 46000,
            "bb_lower": 44000,
            "stoch_k": 50,
            "stoch_d": 48,
            "atr": 800,
        }

        sample_ml = {
            "predicted_price": 46000,
            "current_price": 45000,
            "confidence": 0.75,
        }

        sample_sentiment = {"aggregate_sentiment": 0.2, "confidence": 0.6}

        signal = signal_generator.generate_trading_signal(
            symbol="BTC/USDT",
            current_price=45000,
            technical_indicators=sample_indicators,
            ml_prediction=sample_ml,
            sentiment_data=sample_sentiment,
        )

        print("   ✅ Signal generation successful")
        print(f"   🎯 Signal: {signal['signal_type']}")
        print(f"   🎲 Confidence: {signal['confidence']:.3f}")

    except Exception as e:
        print(f"   ❌ Signal generation error: {e}")

    print("\n" + "=" * 50)
    print("🎉 Component testing completed!")


if __name__ == "__main__":
    asyncio.run(test_all_components())
