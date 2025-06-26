import json
import polars as pl
from typing import Optional


def read_raw_data(
    file_path: str, schema_overrides: Optional[dict] = None
) -> pl.DataFrame:
    """
    Reads the raw movie data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the movie data.
    """
    return pl.read_csv(file_path, schema_overrides=schema_overrides)


def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocesses the movie data by selecting relevant columns and renaming them.

    Columns:
        - 'id': Movie ID
        - 'title': English title of the movie
        - 'original_title': Original title of the movie
        - 'runtime': Runtime in minutes
        - 'overview': Overview of the movie
        - 'genres': List of genres
        - 'keywords': List of keywords
        - 'vote_average': Average vote score
        - 'vote_count': Number of votes

    Where status is 'Released' and the movie has a non-null overview.

    Args:
        df (pl.DataFrame): The raw movie data DataFrame.

    Returns:
        pl.DataFrame: A preprocessed DataFrame with selected and
        renamed columns.
    """
    # Filter for released movies with non-null overview
    filtered_df = df.filter(
        (pl.col("status") == "Released") & (pl.col("overview").is_not_null())
    )

    # Parse genres JSON strings and extract 'name' fields
    filtered_df = filtered_df.with_columns(
        [
            pl.col("genres")
            .str.json_decode()
            .cast(
                pl.List(
                    pl.Struct(
                        [pl.Field("id", pl.Int64), pl.Field("name", pl.Utf8)]
                    )
                ),
                strict=False,
            )
            .list.eval(pl.element().struct.field("name"))
            .map_elements(
                lambda x: json.dumps(
                    x.to_list() if hasattr(x, "to_list") else x
                )
                if x is not None
                else "[]",
                return_dtype=pl.Utf8,
            )
            .alias("genres")
        ]
    )

    # Parse keywords JSON strings and extract 'name' fields
    filtered_df = filtered_df.with_columns(
        [
            pl.col("keywords")
            .str.json_decode()
            .cast(
                pl.List(
                    pl.Struct(
                        [pl.Field("id", pl.Int64), pl.Field("name", pl.Utf8)]
                    )
                ),
                strict=False,
            )
            .list.eval(pl.element().struct.field("name"))
            .map_elements(
                lambda x: json.dumps(
                    x.to_list() if hasattr(x, "to_list") else x
                )
                if x is not None
                else "[]",
                return_dtype=pl.Utf8,
            )
            .alias("keywords")
        ]
    )

    filtered_df = filtered_df.select(
        [
            pl.col("id").alias("movie_id"),
            pl.col("title").alias("english_title"),
            pl.col("original_title"),
            pl.col("runtime"),
            pl.col("overview"),
            pl.col("genres"),
            pl.col("keywords"),  # untouched
            pl.col("vote_average"),
            pl.col("vote_count"),
        ]
    )

    return filtered_df


def save_preprocessed_data(df: pl.DataFrame, output_path: str) -> None:
    """
    Saves the preprocessed movie data to a CSV file.

    Args:
        df (pl.DataFrame): The preprocessed movie data DataFrame.
        output_path (str): The path where the CSV file will be saved.
    """
    df.write_csv(output_path, include_header=True, separator=",")
    print(f"Preprocessed data saved to {output_path}")


def main() -> None:
    raw_data_path = "data/raw/tmdb_5000_movies.csv"
    preprocessed_data_path = "data/preprocessed/tmdb_5000_movies.csv"

    schema_overrides = {
        "runtime": pl.Float32,
    }

    raw_df: pl.DataFrame = read_raw_data(raw_data_path, schema_overrides)
    preprocessed_df: pl.DataFrame = preprocess_data(raw_df)
    save_preprocessed_data(preprocessed_df, preprocessed_data_path)


if __name__ == "__main__":
    main()
