"""
Bulk‑load TMDB 5000 movie metadata into PostgreSQL with SQLAlchemy + Polars.

✓  Uses session.bulk_insert_mappings   → one INSERT per batch, no SELECTs.
✓  Skips rows missing overview / poster / backdrop.
✓  Prints progress every batch.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from db.database import SessionLocal  # your existing sessionmaker
from db.models import Movie  # SQLAlchemy declarative model

PREPROCESSED_CSV = Path("data/preprocessed/tmdb_5000_movies.csv")
BATCH_SIZE = 1_000


def _rows_to_mappings(rows: Iterable[dict]) -> list[dict]:
    """
    Convert Polars row dicts → SQLAlchemy mapping dicts.
    We keep only the model’s column names.
    """
    keep = {
        "movie_id",
        "english_title",
        "original_title",
        "runtime",
        "overview",
        "genres",
        "keywords",
        "vote_average",
        "vote_count",
        "poster_path",
        "backdrop_path",
    }
    return [{k: v for k, v in row.items() if k in keep} for row in rows]

def load_movies_to_db(csv_path: Path = PREPROCESSED_CSV) -> None:
    df = pl.read_csv(csv_path)

    # filter out rows without required fields
    df = df.filter(
        (pl.col("overview").is_not_null())
        & (pl.col("poster_path").is_not_null())
        & (pl.col("backdrop_path").is_not_null())
    )

    total = len(df)
    print(f"Loading {total:,} movies …")

    session = SessionLocal()
    inserted = 0
    try:
        for offset in range(0, total, BATCH_SIZE):
            batch = df[offset : offset + BATCH_SIZE].iter_rows(named=True)
            mappings = _rows_to_mappings(batch)

            session.bulk_insert_mappings(Movie.__mapper__, mappings, render_nulls=True)
            inserted += len(mappings)

            print(
                f"Batch {(offset // BATCH_SIZE) + 1} / {(total - 1) // BATCH_SIZE + 1} "
                f"({inserted:,}/{total:,}) inserted"
            )

        session.commit()
        print(f"✅  Finished: {inserted:,} rows committed.")
    except Exception as e:
        session.rollback()
        print(f"❌  Error loading movies: {e}")
        raise
    finally:
        session.close()


def main() -> None:
    load_dotenv()                  # loads DB creds from .env
    load_movies_to_db()


if __name__ == "__main__":
    main()
