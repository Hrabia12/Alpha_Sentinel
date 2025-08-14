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
                    }
                )
            except Exception as e:
                logging.error(f"Failed to initialize {exchange_name}: {e}")

    async def fetch_historical_data(
        self, symbol: str, timeframe: str = "1h", days: int = 30
    ):
        """Fetch historical OHLCV data"""
        try:
            exchange = self.exchanges["binance"]  # Primary exchange
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, since)

            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
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
            ticker = exchange.fetch_ticker(symbol)

            return {
                "symbol": symbol,
                "exchange": "binance",
                "timestamp": datetime.now(),
                "open": ticker["open"],
                "high": ticker["high"],
                "low": ticker["low"],
                "close": ticker["last"],
                "volume": ticker["baseVolume"],
                "timeframe": "1m",
            }

        except Exception as e:
            logging.error(f"Error fetching real-time data for {symbol}: {e}")
            return None


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
    print(btc_data.head())

    # Test real-time data
    print("\nFetching real-time data...")
    real_time = await collector.fetch_real_time_data("BTC/USDT")
    print(real_time)


if __name__ == "__main__":
    asyncio.run(test_data_collection())
