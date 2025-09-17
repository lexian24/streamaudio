"""
Database configuration and session management for StreamAudio.

This module handles database connections, session management, and 
provides utility functions for database operations.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base

# Database configuration
DATABASE_DIR = Path(__file__).parent.parent / "data"
DATABASE_DIR.mkdir(exist_ok=True)

# SQLite database path
SQLITE_DATABASE_PATH = DATABASE_DIR / "streamaudio.db"
DATABASE_URL = f"sqlite:///{SQLITE_DATABASE_PATH}"
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{SQLITE_DATABASE_PATH}"

# Create sync engine for migrations and setup
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Allow SQLite to be used with multiple threads
    },
    poolclass=StaticPool,
    echo=bool(os.getenv("DEBUG", False))  # Enable SQL logging in debug mode
)

# Create async engine for main application
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
    poolclass=StaticPool,
    echo=bool(os.getenv("DEBUG", False))
)

# Session makers
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


def create_tables():
    """
    Create all database tables.
    This should be called on application startup.
    """
    Base.metadata.create_all(bind=engine)


def get_db_session() -> Session:
    """
    Dependency function to get a database session.
    Used for synchronous database operations.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db_session() -> AsyncSession:
    """
    Dependency function to get an async database session.
    Used for asynchronous database operations.
    """
    async with AsyncSessionLocal() as session:
        yield session


def init_database():
    """
    Initialize the database by creating all tables.
    This function should be called once when setting up the application.
    """
    print(f"Initializing database at: {SQLITE_DATABASE_PATH}")
    create_tables()
    print("Database tables created successfully!")


def reset_database():
    """
    Drop all tables and recreate them.
    WARNING: This will delete all data!
    Use only for development/testing.
    """
    print("WARNING: Resetting database - all data will be lost!")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database reset completed!")


# Database utilities
class DatabaseManager:
    """
    Database manager class for common database operations.
    """
    
    @staticmethod
    def get_database_info():
        """
        Get information about the database.
        """
        return {
            "database_path": str(SQLITE_DATABASE_PATH),
            "database_exists": SQLITE_DATABASE_PATH.exists(),
            "database_size": SQLITE_DATABASE_PATH.stat().st_size if SQLITE_DATABASE_PATH.exists() else 0,
            "database_url": DATABASE_URL,
            "async_database_url": ASYNC_DATABASE_URL
        }
    
    @staticmethod
    def backup_database(backup_path: str = None):
        """
        Create a backup of the database.
        """
        if not SQLITE_DATABASE_PATH.exists():
            raise FileNotFoundError("Database file does not exist")
        
        if backup_path is None:
            backup_path = f"{SQLITE_DATABASE_PATH}.backup"
        
        import shutil
        shutil.copy2(SQLITE_DATABASE_PATH, backup_path)
        return backup_path