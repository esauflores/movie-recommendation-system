import os
import shutil

import kagglehub  # type: ignore


def download_dataset(force_download: bool = True) -> str:
    """
    Downloads the latest version of the TMDB movie metadata dataset.
    Returns the path to the downloaded dataset files.
    """
    path: str = kagglehub.dataset_download("tmdb/tmdb-movie-metadata", force_download=force_download)
    print("Path to dataset files:", path)
    return path


def move_dataset_contents(src_path: str, dest_dir: str) -> None:
    """
    Moves all files and subdirectories from src_path to dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for item in os.listdir(src_path):
        print(f"Moving {item} to {dest_dir}")
        src = os.path.join(src_path, item)
        dest = os.path.join(dest_dir, item)
        if os.path.isdir(src):
            shutil.move(src, dest)
        else:
            shutil.copy(src, dest)


def main() -> None:
    path = download_dataset(force_download=True)
    data_dir = "data/raw"
    move_dataset_contents(path, data_dir)


if __name__ == "__main__":
    main()
