from dotenv import load_dotenv
import polars as pl
from db.database import SessionLocal
from db.models import Movie

PREPROCESSED_CSV = "data/preprocessed/tmdb_5000_movies.csv"


def load_movies_to_db(csv_path: str = PREPROCESSED_CSV) -> None:
    df = pl.read_csv(csv_path)
    session = SessionLocal()
    try:
        for row in df.iter_rows(named=True):
            # if overview is None, or poster_path is None, or backdrop_path is None, skip
            if not row.get("overview") or not row.get("poster_path") or not row.get("backdrop_path"):
                continue

            movie = Movie(
                movie_id=row["movie_id"],
                english_title=row["english_title"],
                original_title=row.get("original_title"),
                runtime=row.get("runtime"),
                overview=row.get("overview"),
                genres=row.get("genres"),
                keywords=row.get("keywords"),
                vote_average=row.get("vote_average"),
                vote_count=row.get("vote_count"),
                poster_path=row.get("poster_path"),
                backdrop_path=row.get("backdrop_path"),
            )
            session.merge(movie)
        session.commit()
        print(f"Loaded {len(df)} movies into the database.")
    except Exception as e:
        session.rollback()
        print(f"Error loading movies: {e}")
    finally:
        session.close()


def main() -> None:
    load_dotenv()
    load_movies_to_db(PREPROCESSED_CSV)


if __name__ == "__main__":
    main()
