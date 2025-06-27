from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from dotenv import load_dotenv
from openai.types import CreateEmbeddingResponse

from db.database import SessionLocal
from db.models import Movie, MovieEmbeddingOpenAI


def generate_embeddings_for_batch(
    batch: list[Movie],
) -> tuple[list[Movie], dict]:
    """
    Generate embeddings for a single batch of movies using concurrent API calls.
    """
    prompts = [
        (
            f"Create an embedding that captures the movie's genre and mood.\n"
            f"Title: {movie.english_title}\n"
            f"Genres: {', '.join(movie.genres) if movie.genres else 'N/A'}\n"
            f"Keywords: {', '.join(movie.keywords) if movie.keywords else 'N/A'}\n"
            f"Overview: {movie.overview or 'No overview available.'}"
        )
        for movie in batch
    ]

    # Define the embedding tasks
    def get_ada_002_embeddings() -> CreateEmbeddingResponse:
        return openai.embeddings.create(
            input=prompts,
            model="text-embedding-ada-002",
        )

    def get_3_small_embeddings() -> CreateEmbeddingResponse:
        return openai.embeddings.create(
            input=prompts,
            model="text-embedding-3-small",
        )

    def get_3_large_embeddings() -> CreateEmbeddingResponse:
        return openai.embeddings.create(
            input=prompts,
            model="text-embedding-3-large",
        )

    # Execute all three embedding API calls concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(get_ada_002_embeddings): "ada_002",
            executor.submit(get_3_small_embeddings): "3_small",
            executor.submit(get_3_large_embeddings): "3_large",
        }

        results = {}
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                results[model_name] = future.result()
            except Exception as e:
                print(f"Error generating {model_name} embeddings: {e}")
                raise e

    return batch, results


def save_embeddings_to_db(batch: list[Movie], embedding_results: dict) -> None:
    """
    Save the generated embeddings to the database using a new session.
    """
    # Create a new session for this batch to avoid concurrency issues
    session = SessionLocal()

    try:
        for j, movie in enumerate(batch):
            try:
                embedding_ada_002 = embedding_results["ada_002"].data[j].embedding
                embedding_3_small = embedding_results["3_small"].data[j].embedding
                embedding_3_large = embedding_results["3_large"].data[j].embedding

                new_embedding = MovieEmbeddingOpenAI(
                    movie_id=movie.movie_id,
                    embedding_ada_002=embedding_ada_002,
                    embedding_3_small=embedding_3_small,
                    embedding_3_large=embedding_3_large,
                )

                session.add(new_embedding)
            except Exception as e:
                print(f"Error processing embeddings for movie {movie.english_title}: {e}")
                raise e

        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def generate_missing_embeddings(batch_size: int = 50, max_workers: int = 3) -> None:
    """
    Finds movies that don't have embeddings and generates embeddings for them using concurrent processing.
    """
    # Create a session just for querying movies without embeddings
    session = SessionLocal()

    try:
        # Find movies without embeddings
        movies_without_embeddings = (
            session.query(Movie)
            .outerjoin(
                MovieEmbeddingOpenAI,
                Movie.movie_id == MovieEmbeddingOpenAI.movie_id,
            )
            .filter(MovieEmbeddingOpenAI.movie_id.is_(None))
            .all()
        )

    finally:
        session.close()

    if not movies_without_embeddings:
        print("All movies already have embeddings.")
        return

    print(f"Found {len(movies_without_embeddings)} movies without embeddings.")

    # Split movies into batches
    batches = [
        movies_without_embeddings[i : i + batch_size] for i in range(0, len(movies_without_embeddings), batch_size)
    ]

    print(f"Processing {len(batches)} batches with up to {max_workers} concurrent batches...")

    # Process batches concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch processing tasks
        future_to_batch = {executor.submit(generate_embeddings_for_batch, batch): i for i, batch in enumerate(batches)}

        # Process completed batches and save to database
        for future in as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                batch, embedding_results = future.result()

                # Save to database (each batch gets its own session to avoid concurrency issues)
                save_embeddings_to_db(batch, embedding_results)

                print(f"✅ Completed batch {batch_index + 1}/{len(batches)} ({len(batch)} movies)")

            except Exception as e:
                print(f"❌ Error processing batch {batch_index + 1}: {e}")
                # Continue with other batches even if one fails

    print("🎉 Embedding generation completed!")


def main() -> None:
    load_dotenv()

    try:
        print("🚀 Starting concurrent embedding generation...")
        # You can adjust batch_size and max_workers based on your OpenAI rate limits
        # and system capabilities
        generate_missing_embeddings(
            batch_size=40,  # Smaller batches for better concurrency
            max_workers=3,  # Conservative to avoid rate limits
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
