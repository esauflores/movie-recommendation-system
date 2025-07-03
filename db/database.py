import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()  # Loads DATABASE_URL from .env


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


def get_database_session() -> tuple[Engine, sessionmaker[Session]]:
    """
    Returns:
    - SQLAlchemy engine
    - sessionmaker bound to that engine
    """

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    # Create engine
    engine = create_engine(database_url)

    # Create configured session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return engine, SessionLocal


# Usage: no stray comma this time
engine, SessionLocal = get_database_session()
