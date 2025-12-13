"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os

# Database URL - using SQLite for simplicity
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smart_contract_analyzer.db")

# Initialize variables that will be set by _initialize_db()
engine = None
SessionLocal = None

# Base class for all models
Base = declarative_base()


def _initialize_db():
    """Initialize database engine and session factory"""
    global engine, SessionLocal
    
    if engine is not None:
        return  # Already initialized
    
    try:
        # Create engine with appropriate settings
        if "sqlite" in DATABASE_URL:
            engine = create_engine(
                DATABASE_URL,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            # For PostgreSQL (and other servers) enable pool_pre_ping to handle
            # stale connections and set a sensible pool size. The DATABASE_URL
            # should be in the format: postgresql://user:password@host:port/dbname
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
            )

        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize database: {e}. Database operations will fail until corrected.")
        engine = None
        SessionLocal = None
        raise


def get_db():
    """Dependency to get database session"""
    _initialize_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    """Get the database engine, initializing if needed"""
    _initialize_db()
    return engine


def create_tables():
    """Create all database tables - skip if database initialization failed"""
    try:
        _initialize_db()
        if engine is not None:
            Base.metadata.create_all(bind=engine)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not create database tables: {e}. Tables will be created on first successful database connection.")
