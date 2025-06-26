from db.database import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, ForeignKey, DateTime
from pgvector.sqlalchemy import VECTOR  # type: ignore
from datetime import datetime


class Movie(Base):
    __tablename__ = "movies"

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    english_title: Mapped[str] = mapped_column(String(255), nullable=False)
    original_title: Mapped[str] = mapped_column(String(255), nullable=True)
    runtime: Mapped[float] = mapped_column(Float, nullable=True)
    overview: Mapped[str] = mapped_column(String, nullable=True)
    genres: Mapped[str] = mapped_column(String, nullable=True)
    keywords: Mapped[str] = mapped_column(String, nullable=True)
    vote_average: Mapped[float] = mapped_column(Float, nullable=True)
    vote_count: Mapped[int] = mapped_column(Integer, nullable=True)
    poster_path: Mapped[str] = mapped_column(String, nullable=True)
    backdrop_path: Mapped[str] = mapped_column(String, nullable=True)


class MovieEmbeddingOpenAI(Base):
    __tablename__ = "movie_embedding_openai"

    movie_id: Mapped[int] = mapped_column(Integer, ForeignKey("movies.movie_id"), primary_key=True)

    embedding_ada_002: Mapped[VECTOR] = mapped_column(VECTOR(1536), nullable=False)  # OpenAI's embedding size is 1536

    embedding_3_small: Mapped[VECTOR] = mapped_column(VECTOR(1536), nullable=True)  # OpenAI's text-embedding-3-small

    embedding_3_large: Mapped[VECTOR] = mapped_column(VECTOR(3072), nullable=True)  # OpenAI's text-embedding-3-large

    embedding_updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now(),
        onupdate=datetime.now(),
    )  # Timestamp when embeddings were last updated

    movie: Mapped[Movie] = relationship("Movie", backref="embedding_openai")
