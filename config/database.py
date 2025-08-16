import os
from supabase import create_client, Client
from dotenv import load_dotenv

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

    def insert_market_data(self, data):
        """Insert market data into database"""
        try:
            # Validate data before insertion
            if not data or not isinstance(data, dict):
                print(f"Invalid data format: {type(data)}")
                return None
                
            # Ensure required fields exist
            required_fields = ["symbol", "timestamp", "close"]
            if not all(field in data for field in required_fields):
                print(f"Missing required fields. Data: {list(data.keys())}")
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
