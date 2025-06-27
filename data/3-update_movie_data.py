from dotenv import load_dotenv
import asyncio
import aiohttp
import os
import time
from typing import List, Any
from datetime import datetime, timedelta
import polars as pl

load_dotenv()


class RateLimiter:
    """Rate limiter to control API requests per minute."""

    def __init__(self, max_requests_per_minute: int = 40):
        self.max_requests = max_requests_per_minute
        self.requests: List[float] = []

    async def acquire(self):
        """Acquire a slot for making a request."""
        now = time.time()

        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]

        # If we're at the limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]) + 0.1  # Add small buffer
            print(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds...")
            await asyncio.sleep(sleep_time)
            return await self.acquire()

        # Record this request
        self.requests.append(now)


async def fetch_movie_data(
    session: aiohttp.ClientSession,
    movie_id: int,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    tmdb_api_key: str,
) -> tuple[int, dict[str, Any] | None]:
    """Fetch movie data from TMDB API with rate limiting and concurrency control."""
    async with semaphore:  # Limit concurrent requests
        await rate_limiter.acquire()  # Rate limiting

        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": tmdb_api_key, "language": "en-US"}

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, params=params, timeout=timeout) as response:
                response.raise_for_status()
                movie_data = await response.json()
                return movie_id, movie_data
        except aiohttp.ClientError as e:
            print(f"Error fetching data for movie ID {movie_id}: {e}")
            return movie_id, None
        except Exception as e:
            print(f"Unexpected error for movie ID {movie_id}: {e}")
            return movie_id, None


async def update_movie_poster_data_async(df: pl.DataFrame, days_threshold: int = 30) -> pl.DataFrame:
    """Update movies with poster data from TMDB API using Polars batch processing."""

    # Filter movies that need updating
    movies_to_update = filter_movies_needing_update(df, days_threshold)

    # If no movies need updating, return original DataFrame
    if len(movies_to_update) == 0:
        print("No movies need updating. Returning original DataFrame.")
        return df

    # Get TMDB API key from environment
    tmdb_api_key = os.getenv("TMDB_API_KEY", None)

    if not tmdb_api_key:
        print("TMDB_API_KEY not found in environment variables")
        return df

    # Extract movie IDs from movies that need updating
    movie_ids = movies_to_update.select("movie_id").to_series().to_list()

    # Configuration
    max_concurrent_requests = 10  # Number of concurrent requests
    max_requests_per_minute = 10000  # Max requests per minute (reduced for safety), adjust as needed
    # currently set to 10000, which is essentially unlimited for testing purposes

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    rate_limiter = RateLimiter(max_requests_per_minute)

    # Store all results
    all_results: dict[int, dict[str, Any]] = {}

    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as http_session:
        # Process movies in batches
        batch_size = 50
        total_batches = (len(movie_ids) + batch_size - 1) // batch_size

        for batch_start in range(0, len(movie_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(movie_ids))
            batch_movie_ids = movie_ids[batch_start:batch_end]

            print(f"Processing batch {batch_start // batch_size + 1}/{total_batches}")

            # Create tasks for concurrent requests
            tasks = [
                fetch_movie_data(http_session, movie_id, semaphore, rate_limiter, tmdb_api_key)
                for movie_id in batch_movie_ids
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Store results
            updated_count = 0
            for movie_id, movie_data in results:
                if movie_data is not None:
                    all_results[movie_id] = {
                        "vote_average": movie_data.get("vote_average"),
                        "vote_count": movie_data.get("vote_count"),
                        "poster_path": movie_data.get("poster_path"),
                        "backdrop_path": movie_data.get("backdrop_path"),
                    }
                    updated_count += 1

            print(f"Batch {batch_start // batch_size + 1} processed: {updated_count} movies updated.")

    # Create update DataFrame from results
    today = get_todays_date()
    update_data: list[dict[str, Any]] = []

    for movie_id in movie_ids:
        if movie_id in all_results:
            api_data = all_results[movie_id]
            row_data: dict[str, Any] = {
                "movie_id": movie_id,
                "vote_average": api_data.get("vote_average"),
                "vote_count": api_data.get("vote_count"),
                "poster_path": api_data.get("poster_path"),
                "backdrop_path": api_data.get("backdrop_path"),
                "updated_at": today,  # Set today's date for updated_at
            }
            update_data.append(row_data)
        else:
            row_data_null: dict[str, Any] = {}
            update_data.append(row_data_null)

    # Create update DataFrame
    update_df = pl.DataFrame(update_data)
    updated_df = df.join(update_df, on="movie_id", how="left", suffix="_new")

    # Use coalesce to prefer new values when available, fallback to original values
    updated_df = updated_df.with_columns(
        [
            pl.coalesce([pl.col("vote_average_new"), pl.col("vote_average")]).alias("vote_average"),
            pl.coalesce([pl.col("vote_count_new"), pl.col("vote_count")]).alias("vote_count"),
            pl.coalesce([pl.col("poster_path_new"), pl.col("poster_path")]).alias("poster_path"),
            pl.coalesce([pl.col("backdrop_path_new"), pl.col("backdrop_path")]).alias("backdrop_path"),
            pl.coalesce([pl.col("updated_at_new"), pl.col("updated_at")]).alias("updated_at"),
        ]
    ).drop(
        [
            "vote_average_new",
            "vote_count_new",
            "poster_path_new",
            "backdrop_path_new",
            "updated_at_new",
        ]
    )

    return updated_df


def update_movie_data(df: pl.DataFrame, days_threshold: int = 30, limit: int = 500) -> pl.DataFrame:
    """
    Synchronous wrapper for the async function.

    Args:
        df: DataFrame with movie data
        days_threshold: Number of days after which data should be refreshed (default: 30)
    """
    return asyncio.run(update_movie_poster_data_async(df, days_threshold))


def filter_movies_needing_update(df: pl.DataFrame, days_threshold: int = 30) -> pl.DataFrame:
    """
    Filter movies that need updating based on updated_at timestamp.

    Args:
        df: DataFrame with movie data
        days_threshold: Number of days after which data should be refreshed

    Returns:
        DataFrame containing only movies that need updating
    """
    # Calculate the cutoff date (today - threshold days)
    cutoff_date = datetime.now() - timedelta(days=days_threshold)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    # Filter movies that need updating:
    # 1. updated_at is None/null
    # 2. updated_at is older than cutoff_date
    movies_to_update = df.filter((pl.col("updated_at").is_null()) | (pl.col("updated_at") < cutoff_str))

    total_needing_update = len(movies_to_update)

    print(f"Total movies: {len(df)}")
    print(f"Total movies needing update (older than {days_threshold} days): {total_needing_update}")
    print(f"Movies selected for this batch: {len(movies_to_update)}")

    return movies_to_update


def get_todays_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def main() -> None:
    df = pl.read_csv("data/preprocessed/tmdb_5000_movies.csv")
    print("Starting movie data update...")

    # Configuration: Update movies older than X days
    DAYS_THRESHOLD = 1  # Update movies older than 1 day for testing purposes

    # The update_movie_data function now handles filtering internally
    updated_df = update_movie_data(df, days_threshold=DAYS_THRESHOLD)

    # Save the updated DataFrame
    output_path = "data/preprocessed/tmdb_5000_movies.csv"
    updated_df.write_csv(output_path)
    print(f"Updated movie data saved to {output_path}")

    print("Movie data update completed.")


if __name__ == "__main__":
    main()
