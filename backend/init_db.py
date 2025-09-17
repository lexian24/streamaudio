#!/usr/bin/env python3
"""
Database initialization script for StreamAudio.

This script sets up the database tables and can be run independently
to initialize or reset the database.

Usage:
    python init_db.py              # Initialize database
    python init_db.py --reset      # Reset database (WARNING: deletes all data)
    python init_db.py --info       # Show database information
"""

import argparse
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent))

from database import init_database, reset_database, DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Initialize StreamAudio database")
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset database (WARNING: deletes all data)"
    )
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show database information"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print("Database Information:")
        info = DatabaseManager.get_database_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    if args.reset:
        print("Resetting database...")
        confirm = input("Are you sure you want to delete all data? (yes/no): ")
        if confirm.lower() == "yes":
            reset_database()
        else:
            print("Database reset cancelled.")
        return
    
    # Default: Initialize database
    print("Initializing StreamAudio database...")
    init_database()


if __name__ == "__main__":
    main()