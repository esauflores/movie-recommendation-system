from dotenv import load_dotenv
from sqlalchemy.orm import Session
from db.database import SessionLocal
from db.models import Movie
import asyncio
import aiohttp  # type: ignore
import os
import time
from typing import List


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
    movie: Movie,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    tmdb_api_key: str,
) -> tuple[Movie, dict | None]:
    """Fetch movie data from TMDB API with rate limiting and concurrency control."""
    async with semaphore:  # Limit concurrent requests
        await rate_limiter.acquire()  # Rate limiting

        url = f"https://api.themoviedb.org/3/movie/{movie.movie_id}"
        params = {"api_key": tmdb_api_key, "language": "en-US"}

        try:
            timeout = aiohttp.ClientTimeout(total=10)  # type: ignore
            async with session.get(url, params=params, timeout=timeout) as response:
                response.raise_for_status()
                movie_data = await response.json()
                return movie, movie_data
        except aiohttp.ClientError as e:
            print(f"Error fetching data for movie ID {movie.movie_id}: {e}")
            return movie, None
        except Exception as e:
            print(f"Unexpected error for movie ID {movie.movie_id}: {e}")
            return movie, None


async def update_movie_poster_data_async(session: Session) -> None:
    """Update movies with poster and backdrop paths from TMDB using async requests."""
    # Get all movies that don't have poster data yet
    movies = session.query(Movie).filter((Movie.poster_path.is_(None))).all()

    print(f"Found {len(movies)} movies without poster data")

    if not movies:
        print("No movies to update")
        return

    # Get TMDB API key from environment
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if not tmdb_api_key:
        print("TMDB_API_KEY not found in environment variables")
        return

    # Configuration
    max_concurrent_requests = 10  # Number of concurrent requests
    max_requests_per_minute = 10000  # Max requests per minute

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    rate_limiter = RateLimiter(max_requests_per_minute)

    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=30)  # type: ignore
    async with aiohttp.ClientSession(timeout=timeout) as http_session:
        # Process movies in batches to avoid overwhelming the database
        batch_size = 50
        for batch_start in range(0, len(movies), batch_size):
            batch_movies = movies[batch_start : batch_start + batch_size]
            print(f"Processing batch {batch_start // batch_size + 1}/{(len(movies) + batch_size - 1) // batch_size}")

            # Create tasks for concurrent requests
            tasks = [
                fetch_movie_data(http_session, movie, semaphore, rate_limiter, tmdb_api_key) for movie in batch_movies
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Update database with results
            updated_count = 0
            for movie, movie_data in results:
                if movie_data is not None:
                    poster_path = movie_data.get("poster_path")
                    backdrop_path = movie_data.get("backdrop_path")

                    if poster_path:
                        movie.poster_path = poster_path
                    if backdrop_path:
                        movie.backdrop_path = backdrop_path
                    updated_count += 1

            # Commit batch updates
            try:
                session.commit()
                print(f"Updated {updated_count}/{len(batch_movies)} movies in this batch")
            except Exception as e:
                print(f"Error committing batch: {e}")
                session.rollback()


def update_movie_poster_data(session: Session) -> None:
    """Synchronous wrapper for the async function."""
    asyncio.run(update_movie_poster_data_async(session))


def main() -> None:
    load_dotenv()

    session: Session = SessionLocal()

    try:
        update_movie_poster_data(session)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
