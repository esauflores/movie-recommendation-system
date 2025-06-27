from collections.abc import Callable
from enum import Enum

import openai
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import func, select
from sqlalchemy.orm import Mapped
from sqlalchemy.sql import ColumnElement

from db.database import SessionLocal
from db.models import Movie, MovieEmbeddingOpenAI


class EmbeddingModel(Enum):
    ADA_002 = ("text-embedding-ada-002", MovieEmbeddingOpenAI.embedding_ada_002)
    SMALL_3 = ("text-embedding-3-small", MovieEmbeddingOpenAI.embedding_3_small)
    LARGE_3 = ("text-embedding-3-large", MovieEmbeddingOpenAI.embedding_3_large)

    def __init__(self, model_name: str, db_column: Mapped[VECTOR]) -> None:
        self.model_name = model_name
        self.db_column = db_column


def score_v1(embedding: list[float], embedding_model: EmbeddingModel) -> ColumnElement[float]:
    """
    Return cosine similarity score: 1 - cosine distance
    between the query embedding and stored movie embeddings.

    Equation:
        score_metric =
        1 - cosine_distance
    """
    similarity = embedding_model.db_column.cosine_distance(embedding)  # type: ignore
    score_metric = 1 - similarity

    return score_metric  # type: ignore


def score_v2(embedding: list[float], embedding_model: EmbeddingModel) -> ColumnElement[float]:
    """
    Compute weighted score combining cosine similarity, vote average, and log vote count.

    Equation:
        score_metric =
        0.8 * (1 - cosine_distance)
        + 0.2 * (vote_average / 10)
        + 0.1 * log(1 + vote_count)
    """
    similarity = embedding_model.db_column.cosine_distance(embedding)  # type: ignore
    similarity_score = 1 - similarity

    score_metric = 0.8 * similarity_score + 0.2 * (Movie.vote_average / 10.0) + 0.1 * func.log(1 + Movie.vote_count)

    return score_metric  # type: ignore


def score_v3(embedding: list[float], embedding_model: EmbeddingModel) -> ColumnElement[float]:
    """
    Compute weighted score using L2 similarity, capped log vote count, and vote average.
    Use least to cap the maximum log vote count to 10.

    Equation:
        score_metric =
        0.9 * (1 - cosine_distance)
        + 0.07 * (vote_average / 10)
        + 0.03 * least(10, log(1 + vote_count))
    """
    similarity = embedding_model.db_column.cosine_distance(embedding)  # type: ignore
    similarity_score = 1 - similarity

    score_metric = (
        0.9 * similarity_score
        + 0.07 * (Movie.vote_average / 10.0)
        + 0.03 * func.least(10, func.log(1 + Movie.vote_count))
    )

    return score_metric  # type: ignore


class ScoreMetricVersion(Enum):
    V1 = ("v1", score_v1)
    V2 = ("v2", score_v2)
    V3 = ("v3", score_v3)

    def __init__(self, version: str, score_function: Callable) -> None:
        self.version = version
        self.score_function = score_function


EMBEDDING_MODEL = EmbeddingModel.LARGE_3
SCORE_METRIC_VERSION = ScoreMetricVersion.V3


def get_recommendations(
    prompt: str,
    page: int = 1,
    per_page: int = 10,
    embedding_model: EmbeddingModel = EMBEDDING_MODEL,
    score_metric_version: ScoreMetricVersion = SCORE_METRIC_VERSION,
) -> list[Movie]:
    """Get movie recommendations based on user prompt using embeddings."""
    offset = (page - 1) * per_page

    # Step 1: Embed the user prompt
    response = openai.embeddings.create(input=prompt, model=embedding_model.model_name)

    embedding: list[float] = response.data[0].embedding

    score_metric = score_metric_version.score_function(embedding, embedding_model)

    stmt = (
        select(Movie)
        .join(
            MovieEmbeddingOpenAI,
            Movie.movie_id == MovieEmbeddingOpenAI.movie_id,
        )
        .order_by(score_metric.desc())  # pyrefly: ignore
        .offset(offset)
        .limit(per_page)
    )

    session = SessionLocal()

    try:
        results = session.execute(stmt).scalars().all()
        return list(results)

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_movie_by_id(movie_id: int) -> Movie | None:
    """Get a specific movie by its ID."""
    session = SessionLocal()
    try:
        stmt = select(Movie).where(Movie.movie_id == movie_id)
        result = session.execute(stmt).scalar_one_or_none()
        return result  # type: ignore
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_similar_movies(
    movie_id: int,
    page: int = 1,
    per_page: int = 10,
    embedding_model: EmbeddingModel = EMBEDDING_MODEL,
    score_metric_version: ScoreMetricVersion = SCORE_METRIC_VERSION,
) -> list[Movie]:
    """Get movies similar to the given movie based on embeddings."""
    offset = (page - 1) * per_page
    session = SessionLocal()
    try:
        # First, get the embedding of the target movie
        target_stmt = select(embedding_model.db_column).where(MovieEmbeddingOpenAI.movie_id == movie_id)
        target_embedding = session.execute(target_stmt).scalar_one_or_none()

        if target_embedding is None:
            return []

        score_metric = score_metric_version.score_function(target_embedding, embedding_model)

        stmt = (
            select(Movie)
            .join(
                MovieEmbeddingOpenAI,
                Movie.movie_id == MovieEmbeddingOpenAI.movie_id,
            )
            .where(Movie.movie_id != movie_id)  # Exclude the original movie
            .order_by(score_metric.desc())  # pyrefly: ignore
            .offset(offset)
            .limit(per_page)
        )

        results = session.execute(stmt).scalars().all()
        return list(results)

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
