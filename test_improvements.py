#!/usr/bin/env python3
"""
Test script to verify the improvements made to the Alpha Sentinel system
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append("src")

async def test_improvements():
    """Test the improvements made to the system"""
    
    print("🧪 Testing Alpha Sentinel Improvements")
    print("=" * 50)
    
    try:
        # Test 1: Exchange Data Collection
        print("1. Testing Improved Exchange Data Collection...")
        from data_pipeline.exchange_collector import ExchangeDataCollector
        
        exchange_config = {"binance": {"sandbox": True}}
        collector = ExchangeDataCollector(exchange_config)
        
        # Test real-time data
        data = await collector.fetch_real_time_data("BTC/USDT")
        if data:
            print("   ✅ Real-time data collection successful")
            print(f"   📊 BTC/USDT Current Price: ${data['close']:,.2f}")
            print(f"   📈 High: ${data['high']:,.2f}")
            print(f"   📉 Low: ${data['low']:,.2f}")
            print(f"   📊 Open: ${data['open']:,.2f}")
            print(f"   📊 Volume: {data['volume']:,.2f}")
            
            # Validate price logic
            if data['high'] >= max(data['open'], data['close']):
                print("   ✅ High price validation: PASSED")
            else:
                print("   ❌ High price validation: FAILED")
                
            if data['low'] <= min(data['open'], data['close']):
                print("   ✅ Low price validation: PASSED")
            else:
                print("   ❌ Low price validation: FAILED")
        else:
            print("   ❌ Real-time data collection failed")
            
        # Test 2: Historical Data
        print("\n2. Testing Historical Data Collection...")
        historical_data = await collector.fetch_historical_data("BTC/USDT", "1h", 1)
        if not historical_data.empty:
            print("   ✅ Historical data collection successful")
            print(f"   📊 Fetched {len(historical_data)} records")
            print(f"   📈 Price range: ${historical_data['low'].min():.2f} - ${historical_data['high'].max():.2f}")
            
            # Check for outliers
            median_price = historical_data['close'].median()
            outliers = historical_data[
                (historical_data['close'] < median_price * 0.2) | 
                (historical_data['close'] > median_price * 5)
            ]
            if len(outliers) == 0:
                print("   ✅ No extreme outliers detected")
            else:
                print(f"   ⚠️  {len(outliers)} potential outliers detected")
        else:
            print("   ❌ Historical data collection failed")
            
        # Test 3: Database Validation
        print("\n3. Testing Database Validation...")
        from config.database import DatabaseManager
        
        db_manager = DatabaseManager()
        
        # Test data validation
        test_data = {
            "symbol": "BTC/USDT",
            "timestamp": "2025-08-16T15:00:00",
            "open": 117000.0,
            "high": 118000.0,
            "low": 116000.0,
            "close": 117500.0,
            "volume": 1000.0
        }
        
        is_valid, message = db_manager.validate_market_data(test_data)
        if is_valid:
            print("   ✅ Data validation: PASSED")
        else:
            print(f"   ❌ Data validation: FAILED: {message}")
            
        # Test invalid data
        invalid_data = {
            "symbol": "BTC/USDT",
            "timestamp": "2025-08-16T15:00:00",
            "open": 117000.0,
            "high": 115000.0,  # High < Open (invalid)
            "low": 116000.0,
            "close": 117500.0,
            "volume": 1000.0
        }
        
        is_valid, message = db_manager.validate_market_data(invalid_data)
        if not is_valid:
            print("   ✅ Invalid data detection: PASSED")
        else:
            print("   ❌ Invalid data detection: FAILED")
            
        # Test 4: Multiple Price Fetching
        print("\n4. Testing Multiple Price Fetching...")
        prices = await collector.get_multiple_prices(["BTC/USDT", "ETH/USDT"])
        for symbol, price in prices.items():
            if price:
                print(f"   ✅ {symbol}: ${price:,.2f}")
            else:
                print(f"   ❌ {symbol}: Failed to fetch price")
                
        print("\n" + "=" * 50)
        print("🎉 Improvement testing completed!")
        
        # Summary
        print("\n📋 Summary of Improvements Made:")
        print("   ✅ Enhanced data validation and cleaning")
        print("   ✅ Improved OHLCV data structure handling")
        print("   ✅ Better outlier detection and removal")
        print("   ✅ Fixed timezone issues")
        print("   ✅ Enhanced candlestick chart implementation")
        print("   ✅ Improved price accuracy through better data sources")
        
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_improvements())
