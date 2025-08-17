import os
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd

# DEBUG: Check if .env file is found
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
print(f"Looking for .env file at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

# Load environment variables
load_dotenv(env_path)

class DatabaseManager:
    def __init__(self):
        # Temporary: Set environment variables directly
        os.environ["SUPABASE_URL"] = "https://amhpflwemffmokxuurpk.supabase.co"
        os.environ["SUPABASE_ANON_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtaHBmbHdlbWZmbW9reHV1cnBrIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTExNjA1MSwiZXhwIjoyMDcwNjkyMDUxfQ.xecfGUNWwRQdPxNLXO2O6FpqXEDuRaHLXiYcBpKxJM0"
        
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        # DEBUG: Print the key (first 20 chars) to verify it's loaded
        print(f"Supabase URL: {self.supabase_url}")
        print(f"Supabase Key: {self.supabase_key[:20]}..." if self.supabase_key else "NO KEY FOUND")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        self.last_error = None

    def validate_market_data(self, data):
        """Validate market data before insertion"""
        if not data or not isinstance(data, dict):
            return False, "Invalid data format"
            
        # Ensure required fields exist
        required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        if not all(field in data for field in required_fields):
            return False, f"Missing required fields. Found: {list(data.keys())}"
        
        # Validate price data
        try:
            open_price = float(data["open"])
            high_price = float(data["high"])
            low_price = float(data["low"])
            close_price = float(data["close"])
            volume = float(data["volume"])
        except (ValueError, TypeError):
            return False, "Invalid numeric values in price data"
        
        # Validate price logic
        if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
            return False, "All prices must be positive"
        
        if volume < 0:
            return False, "Volume must be non-negative"
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        if high_price < max(open_price, close_price):
            return False, "High price must be >= max(open, close)"
        
        if low_price > min(open_price, close_price):
            return False, "Low price must be <= min(open, close)"
        
        # Check for extreme outliers (prices that are more than 10x the median-like value)
        avg_price = (open_price + high_price + low_price + close_price) / 4
        if any(abs(price - avg_price) > avg_price * 10 for price in [open_price, high_price, low_price, close_price]):
            return False, "Price values appear to be extreme outliers"
        
        return True, "Data is valid"

    def insert_market_data(self, data):
        """Insert market data into database"""
        try:
            # Validate data before insertion
            is_valid, message = self.validate_market_data(data)
            if not is_valid:
                print(f"Data validation failed: {message}")
                return None
                
            # Check if data already exists to avoid duplicates
            existing = self.client.table("market_data").select("id").eq("symbol", data["symbol"]).eq("timestamp", data["timestamp"]).execute()
            if existing.data:
                print(f"Data already exists for {data['symbol']} at {data['timestamp']}")
                return existing.data[0]
                
            result = self.client.table("market_data").insert(data).execute()
            return result
        except Exception as e:
            self.last_error = str(e)
            print(f"Error inserting market data: {e}")
            # Log more details for debugging
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None

    def insert_prediction(self, data):
        """Insert ML prediction into database"""
        try:
            result = self.client.table("ml_predictions").insert(data).execute()
            return result
        except Exception as e:
            print(f"Error inserting prediction: {e}")
            return None

    def get_recent_data(self, symbol, limit=100):
        """Get recent market data for a symbol"""
        try:
            result = (
                self.client.table("market_data")
                .select("*")
                .eq("symbol", symbol)
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def get_clean_market_data(self, symbol, limit=100):
        """Get recent market data with additional cleaning"""
        try:
            raw_data = self.get_recent_data(symbol, limit)
            if not raw_data:
                return []
            
            # Additional cleaning of retrieved data
            cleaned_data = []
            for record in raw_data:
                try:
                    # Ensure numeric values
                    record["open"] = float(record["open"])
                    record["high"] = float(record["high"])
                    record["low"] = float(record["low"])
                    record["close"] = float(record["close"])
                    record["volume"] = float(record["volume"])
                    
                    # Validate price logic
                    if (record["high"] >= max(record["open"], record["close"]) and 
                        record["low"] <= min(record["open"], record["close"]) and
                        all(price > 0 for price in [record["open"], record["high"], record["low"], record["close"]])):
                        cleaned_data.append(record)
                    else:
                        print(f"Skipping invalid record for {symbol}: {record}")
                        
                except (ValueError, TypeError) as e:
                    print(f"Error processing record for {symbol}: {e}")
                    continue
            
            return cleaned_data
            
        except Exception as e:
            print(f"Error getting clean market data: {e}")
            return []

    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old market data to prevent database bloat"""
        try:
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            # Delete old market data
            result = self.client.table("market_data").delete().lt("timestamp", cutoff_date).execute()
            print(f"Cleaned up {len(result.data) if result.data else 0} old records")
            
            return True
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
            return False

    def cleanup_corrupted_data(self):
        """Clean up corrupted or extreme outlier data"""
        try:
            print("🧹 Starting corrupted data cleanup...")
            
            # Get all market data
            result = self.client.table("market_data").select("*").execute()
            if not result.data:
                print("No market data found to clean")
                return True
            
            df = pd.DataFrame(result.data)
            original_count = len(df)
            
            # Convert to numeric and handle errors
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])
            
            # Remove extreme outliers using IQR method
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    # Conservative bounds
                    lower_bound = q1 - 2 * iqr
                    upper_bound = q3 + 2 * iqr
                    
                    # Additional sanity check
                    median_price = df[col].median()
                    if median_price > 0:
                        lower_bound = max(lower_bound, median_price * 0.3)
                        upper_bound = min(upper_bound, median_price * 3.0)
                    
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Ensure price logic is correct
            df = df[
                (df["high"] >= df[["open", "close"]].max(axis=1)) &
                (df["low"] <= df[["open", "close"]].min(axis=1))
            ]
            
            # Remove unrealistic price movements (>50% in one period)
            if len(df) > 1:
                df = df.sort_values("timestamp")
                df["price_change_pct"] = df["close"].pct_change().abs() * 100
                df = df[df["price_change_pct"] <= 50]
                df = df.drop("price_change_pct", axis=1)
            
            cleaned_count = len(df)
            removed_count = original_count - cleaned_count
            
            print(f"Data cleanup: {removed_count} corrupted records removed")
            print(f"Remaining records: {cleaned_count}")
            
            if removed_count > 0:
                # Get the IDs of records to keep
                keep_ids = set(df["id"].tolist())
                
                # Delete the corrupted records (those not in keep_ids)
                for record in result.data:
                    if record["id"] not in keep_ids:
                        try:
                            self.client.table("market_data").delete().eq("id", record["id"]).execute()
                        except Exception as e:
                            print(f"Error deleting corrupted record {record['id']}: {e}")
                
                print(f"✅ Cleaned up {removed_count} corrupted records")
            
            return True
            
        except Exception as e:
            print(f"Error during corrupted data cleanup: {e}")
            return False

    def get_data_quality_report(self):
        """Generate a report on data quality"""
        try:
            result = self.client.table("market_data").select("*").execute()
            if not result.data:
                return "No market data found"
            
            df = pd.DataFrame(result.data)
            
            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Count NaN values
            nan_counts = df[["open", "high", "low", "close", "volume"]].isna().sum()
            
            # Check for extreme outliers
            outlier_counts = {}
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 2 * iqr
                    upper_bound = q3 + 2 * iqr
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_counts[col] = len(outliers)
            
            # Check price logic violations
            logic_violations = len(df[
                ~((df["high"] >= df[["open", "close"]].max(axis=1)) &
                  (df["low"] <= df[["open", "close"]].min(axis=1)))
            ])
            
            report = f"""
📊 Data Quality Report
======================
Total Records: {len(df)}
NaN Values: {nan_counts.to_dict()}
Extreme Outliers: {outlier_counts}
Price Logic Violations: {logic_violations}
Data Quality Score: {((len(df) - sum(nan_counts) - sum(outlier_counts.values()) - logic_violations) / len(df) * 100):.1f}%
            """
            
            return report
            
        except Exception as e:
            return f"Error generating report: {e}"
