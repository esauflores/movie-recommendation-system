from dotenv import load_dotenv
from sqlalchemy.orm import Session
from db.database import SessionLocal
from db.models import Movie, MovieEmbeddingOpenAI


def remove_movies_without_poster(session: Session, batch_size: int = 50):
    """
    Finds movies that don't have poster or back and removes them from the database.
    """

    # Find movies without poster
    movies_without_poster = (
        session.query(Movie)
        .filter(
            (Movie.poster_path.is_(None)) | (Movie.poster_path == ""),
        )
        .all()
    )

    if not movies_without_poster:
        print("No movies without poster found.")
        return

    print(f"Found {len(movies_without_poster)} movies without poster.")

    for i in range(0, len(movies_without_poster), batch_size):
        batch = movies_without_poster[i : i + batch_size]

        for movie in batch:
            # Remove movie embeddings
            session.query(MovieEmbeddingOpenAI).filter(MovieEmbeddingOpenAI.movie_id == movie.movie_id).delete()

            # Remove movie
            session.delete(movie)

        session.commit()
        print(f"Removed batch {i // batch_size + 1} of {len(movies_without_poster) // batch_size + 1}")


def remove_movies_without_backdrop(session: Session, batch_size: int = 50):
    """
    Finds movies that don't have backdrop and removes them from the database.
    """

    # Find movies without backdrop
    movies_without_backdrop = (
        session.query(Movie)
        .filter(
            (Movie.backdrop_path.is_(None)) | (Movie.backdrop_path == ""),
        )
        .all()
    )

    if not movies_without_backdrop:
        print("No movies without backdrop found.")
        return

    print(f"Found {len(movies_without_backdrop)} movies without backdrop.")

    for i in range(0, len(movies_without_backdrop), batch_size):
        batch = movies_without_backdrop[i : i + batch_size]

        for movie in batch:
            # Remove movie embeddings
            session.query(MovieEmbeddingOpenAI).filter(MovieEmbeddingOpenAI.movie_id == movie.movie_id).delete()

            # Remove movie
            session.delete(movie)

        session.commit()
        print(f"Removed batch {i // batch_size + 1} of {len(movies_without_backdrop) // batch_size + 1}")


def main() -> None:
    load_dotenv()

    session: Session = SessionLocal()

    try:
        remove_movies_without_poster(session)
        remove_movies_without_backdrop(session)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
