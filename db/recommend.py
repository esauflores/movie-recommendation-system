import openai
from sqlalchemy import func, select
from db.database import SessionLocal
from db.models import Movie, MovieEmbeddingOpenAI

EMBEDDING_MODELS = {
    "text-embedding-ada-002": MovieEmbeddingOpenAI.embedding_ada_002,
    "text-embedding-3-small": MovieEmbeddingOpenAI.embedding_3_small,
    "text-embedding-3-large": MovieEmbeddingOpenAI.embedding_3_large,
}


def calculate_score_metric(
    embedding: list[float],
    include_vote_count: bool = True,
    embedding_model: str = "text-embedding-3-large",
):
    """Calculate the score metric based on the embedding and vote count."""
    distance_expr = EMBEDDING_MODELS[embedding_model].l2_distance(embedding)
    similarity_score = 1 - distance_expr

    score_metric = (
        0.9 * similarity_score
        + 0.07 * (Movie.vote_average / 10.0)
        + 0.03 * func.least(10, func.log(1 + Movie.vote_count))
    )

    return score_metric


def get_recommendations(
    prompt: str,
    page: int = 1,
    per_page: int = 10,
    embedding_model: str = "text-embedding-3-large",
) -> list[Movie]:
    offset = (page - 1) * per_page

    # Step 1: Embed the user prompt
    response = openai.embeddings.create(input=prompt, model=embedding_model)

    embedding: list[float] = response.data[0].embedding

    score_metric = calculate_score_metric(
        embedding,
        include_vote_count=True,
        embedding_model=embedding_model,
    )

    stmt = (
        select(Movie)
        .join(
            MovieEmbeddingOpenAI,
            Movie.movie_id == MovieEmbeddingOpenAI.movie_id,
        )
        .order_by(score_metric.desc())
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

    return []


def get_movie_by_id(movie_id: int) -> Movie | None:
    """Get a specific movie by its ID."""
    session = SessionLocal()
    try:
        stmt = select(Movie).where(Movie.movie_id == movie_id)
        result = session.execute(stmt).scalar_one_or_none()
        return result
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_similar_movies(
    movie_id: int,
    limit: int = 10,
    embedding_model: str = "text-embedding-3-large",
) -> list[Movie]:
    """Get movies similar to the given movie based on embeddings."""
    session = SessionLocal()
    try:
        # First, get the embedding of the target movie
        target_stmt = select(EMBEDDING_MODELS[embedding_model]).where(
            MovieEmbeddingOpenAI.movie_id == movie_id
        )
        target_embedding = session.execute(target_stmt).scalar_one_or_none()

        if target_embedding is None:
            return []

        # Find similar movies using embedding similarity
        score_metric = calculate_score_metric(
            target_embedding,
            include_vote_count=False,
            embedding_model=embedding_model,
        )

        stmt = (
            select(Movie)
            .join(
                MovieEmbeddingOpenAI,
                Movie.movie_id == MovieEmbeddingOpenAI.movie_id,
            )
            .where(Movie.movie_id != movie_id)  # Exclude the original movie
            .order_by(score_metric.desc())
            .limit(limit)
        )

        results = session.execute(stmt).scalars().all()
        return list(results)

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    return []
