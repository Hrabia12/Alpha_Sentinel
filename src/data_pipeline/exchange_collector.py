import ccxt
import asyncio
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict
import logging


class ExchangeDataCollector:
    def __init__(self, exchanges_config: Dict):
        self.exchanges = {}
        self.target_symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "DOGE/USDT",
        ]

        # Initialize exchanges
        for exchange_name, config in exchanges_config.items():
            try:
                exchange_class = getattr(ccxt, exchange_name)
                self.exchanges[exchange_name] = exchange_class(
                    {
                        "apiKey": config.get("api_key"),
                        "secret": config.get("api_secret"),
                        "sandbox": config.get(
                            "sandbox", True
                        ),  # Use sandbox for testing
                        "enableRateLimit": True,
                        "options": {
                            "defaultType": "spot",  # Ensure we're getting spot market data
                        }
                    }
                )
            except Exception as e:
                logging.error(f"Failed to initialize {exchange_name}: {e}")

    async def fetch_historical_data(
        self, symbol: str, timeframe: str = "1h", days: int = 30
    ):
        
        try:
            exchange = self.exchanges["binance"]  # Primary exchange
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, since)

            if not ohlcv_data:
                logging.warning(f"No historical data received for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            
            # Clean and validate the data
            df = self._clean_ohlcv_data(df)
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = symbol
            df["exchange"] = "binance"
            df["timeframe"] = timeframe

            return df

        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_real_time_data(self, symbol: str):
        """Fetch real-time ticker data"""
        try:
            exchange = self.exchanges["binance"]
            
            # Get both ticker and recent OHLCV for more accurate data
            ticker = exchange.fetch_ticker(symbol)
            
            # Also get the most recent candlestick for more accurate OHLC data
            recent_ohlcv = exchange.fetch_ohlcv(symbol, "1m", limit=1)
            
            if recent_ohlcv and len(recent_ohlcv) > 0:
                # Use the most recent OHLCV data for more accuracy
                latest_candle = recent_ohlcv[-1]
                open_price = latest_candle[1]
                high_price = latest_candle[2]
                low_price = latest_candle[3]
                close_price = latest_candle[4]
                volume = latest_candle[5]
            else:
                # Fallback to ticker data
                open_price = ticker["open"]
                high_price = ticker["high"]
                low_price = ticker["low"]
                close_price = ticker["last"]
                volume = ticker["baseVolume"]

            return {
                "symbol": symbol,
                "exchange": "binance",
                "timestamp": datetime.now(),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "timeframe": "1m",
            }

        except Exception as e:
            logging.error(f"Error fetching real-time data for {symbol}: {e}")
            return None

    def _clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data"""
        if df.empty:
            return df
        
        # Remove rows with invalid data
        df = df[
            (df["open"] > 0) & 
            (df["high"] > 0) & 
            (df["low"] > 0) & 
            (df["close"] > 0) &
            (df["volume"] >= 0)
        ]
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        
        # Remove extreme outliers (prices that are more than 5x the median)
        for col in ["open", "high", "low", "close"]:
            median_price = df[col].median()
            if median_price > 0:
                df = df[
                    (df[col] >= median_price * 0.2) & 
                    (df[col] <= median_price * 5)
                ]
        
        return df

    async def get_current_price(self, symbol: str) -> float:
        """Get just the current price for a symbol"""
        try:
            exchange = self.exchanges["binance"]
            ticker = exchange.fetch_ticker(symbol)
            return ticker["last"]
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {e}")
            return None

    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols efficiently"""
        try:
            exchange = self.exchanges["binance"]
            tickers = exchange.fetch_tickers(symbols)
            
            prices = {}
            for symbol in symbols:
                if symbol in tickers:
                    prices[symbol] = tickers[symbol]["last"]
                else:
                    prices[symbol] = None
                    
            return prices
        except Exception as e:
            logging.error(f"Error fetching multiple prices: {e}")
            return {symbol: None for symbol in symbols}


# Example usage and testing
async def test_data_collection():
    config = {
        "binance": {
            "api_key": None,  # Not needed for public market data
            "api_secret": None,
            "sandbox": True,
        }
    }

    collector = ExchangeDataCollector(config)

    # Test historical data
    print("Fetching historical data for BTC/USDT...")
    btc_data = await collector.fetch_historical_data("BTC/USDT", "1h", 7)
    print(f"Fetched {len(btc_data)} records")
    if not btc_data.empty:
        print("Sample data:")
        print(btc_data.head())
        print(f"Price range: ${btc_data['low'].min():.2f} - ${btc_data['high'].max():.2f}")

    # Test real-time data
    print("\nFetching real-time data...")
    real_time = await collector.fetch_real_time_data("BTC/USDT")
    if real_time:
        print(f"Current BTC price: ${real_time['close']:,.2f}")
        print(f"24h high: ${real_time['high']:,.2f}")
        print(f"24h low: ${real_time['low']:,.2f}")
        print(f"Volume: {real_time['volume']:,.2f}")

    # Test multiple prices
    print("\nFetching multiple prices...")
    prices = await collector.get_multiple_prices(["BTC/USDT", "ETH/USDT"])
    for symbol, price in prices.items():
        if price:
            print(f"{symbol}: ${price:,.2f}")


if __name__ == "__main__":
    asyncio.run(test_data_collection())
