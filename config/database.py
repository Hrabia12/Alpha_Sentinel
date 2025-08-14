import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.client: Client = create_client(self.supabase_url, self.supabase_key)

    def insert_market_data(self, data):
        """Insert market data into database"""
        try:
            result = self.client.table("market_data").insert(data).execute()
            return result
        except Exception as e:
            print(f"Error inserting market data: {e}")
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
