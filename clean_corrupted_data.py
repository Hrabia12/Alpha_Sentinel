#!/usr/bin/env python3
"""
Script to clean corrupted data from the Alpha Sentinel database
This will remove extreme outliers and fix the chart display issues
"""

import sys
import os

# Add src to path
sys.path.append("src")

async def clean_corrupted_data():
    """Clean corrupted data from the database"""
    
    print("🧹 Alpha Sentinel Data Cleanup Tool")
    print("=" * 50)
    
    try:
        from config.database import DatabaseManager
        
        # Initialize database manager
        print("1. Connecting to database...")
        db_manager = DatabaseManager()
        print("   ✅ Database connection successful")
        
        # Generate data quality report before cleanup
        print("\n2. Current Data Quality Report:")
        report = db_manager.get_data_quality_report()
        print(report)
        
        # Ask for confirmation
        print("\n⚠️  WARNING: This will permanently delete corrupted data!")
        print("   This includes extreme outliers and invalid price records.")
        
        response = input("\nDo you want to proceed with cleanup? (yes/no): ").lower().strip()
        
        if response not in ['yes', 'y']:
            print("❌ Cleanup cancelled by user")
            return
        
        # Perform cleanup
        print("\n3. Starting data cleanup...")
        success = db_manager.cleanup_corrupted_data()
        
        if success:
            print("   ✅ Data cleanup completed successfully")
            
            # Generate new data quality report
            print("\n4. New Data Quality Report:")
            new_report = db_manager.get_data_quality_report()
            print(new_report)
            
            print("\n🎉 Cleanup completed! Your charts should now display properly.")
            print("   Restart your dashboard to see the improvements.")
            
        else:
            print("   ❌ Data cleanup failed")
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(clean_corrupted_data())
