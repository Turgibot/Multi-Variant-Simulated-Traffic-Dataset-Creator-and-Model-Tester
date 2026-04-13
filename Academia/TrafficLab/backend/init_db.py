#!/usr/bin/env python3
"""
Database initialization script
Creates all necessary tables for the TrafficLab application
"""

from models.database import create_tables, engine
import os

def main():
    print("🗄️ Initializing TrafficLab database...")
    
    # Create all tables
    try:
        create_tables()
        print("✅ Database tables created successfully!")
        
        # Test database connection
        from models.database import get_db
        db = next(get_db())
        print("✅ Database connection successful!")
        
        # Show table info
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"📊 Created tables: {tables}")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Database initialization complete!")
    else:
        print("💥 Database initialization failed!")
