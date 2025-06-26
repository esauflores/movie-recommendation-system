import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from sqlalchemy.engine import Engine

load_dotenv()


class Base(DeclarativeBase):
    pass


def get_database_session() -> tuple[Engine, sessionmaker[Session]]:
    """
    Loads environment variables and returns:
    - SQLAlchemy engine
    - sessionmaker
    - declarative base
    """

    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")

    if not all([db_host, db_port, db_name, db_user, db_pass]):
        raise ValueError("Database connection env vars are not fully set.")

    database_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    # Create engine
    engine = create_engine(database_url)

    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Base class for models
    return engine, SessionLocal


engine, SessionLocal = get_database_session()
