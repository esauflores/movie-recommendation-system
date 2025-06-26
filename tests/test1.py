from dotenv import load_dotenv
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from db.database import SessionLocal
from db.models import Movie, MovieEmbeddingOpenAI
import openai


def recommend_movies(
    session: Session,
    user_query: str,
    top_k: int = 10,
    embedding_model: str = "text-embedding-ada-002",
):
    # 1. Embed the user query
    response = openai.embeddings.create(
        input=user_query,
        model=embedding_model,
    )

    query_embedding = response.data[0].embedding

    # 2. Create the distance expression
    # The <=> operator returns cosine distance (0 = identical, 2 = opposite)
    distance_expr = MovieEmbeddingOpenAI.embedding_ada_002.l2_distance(
        query_embedding
    )

    # Convert distance to similarity (1 - distance/2 for cosine distance)
    # Or simply use (1 - distance) if your distance is already normalized
    similarity_score = 1 - distance_expr

    # 3. Create combined scoring expression
    combined_score = (
        0.7 * similarity_score
        + 0.2 * (Movie.vote_average / 10.0)
        + 0.1 * func.log(1 + Movie.vote_count)
    )

    # 2. Query top-k closest movies by similarity (using cosine distance)
    stmt = (
        select(Movie)
        .join(
            MovieEmbeddingOpenAI,
            Movie.movie_id == MovieEmbeddingOpenAI.movie_id,
        )
        .order_by(combined_score.desc())
        .limit(top_k)
    )

    results = session.execute(stmt).scalars().all()

    # 3. Return recommended movies
    return results


def main():
    load_dotenv()

    session = SessionLocal()

    prompt = "I want to watch a movie about superheroes with a lot of action."

    print(f"Generating recommendations for prompt: {prompt}")

    try:
        recommendations = recommend_movies(session, prompt, top_k=10)
        print("Recommended Movies:")
        for movie in recommendations:
            print(f"- {movie.english_title} (ID: {movie.movie_id})")
    except Exception as e:
        print(f"Error during recommendation: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
