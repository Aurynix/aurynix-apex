"""Download the raw dataset from Kaggle into data/raw/.

Run `python -m apex.data.download` (or `make data-download`). Kaggle file names
are mapped to the names in config.json → files, so the rest of the code never
depends on how the publisher named them.
"""

import shutil
from pathlib import Path

from apex.config import load_config, path


def download_raw() -> list[Path]:
    """Download the Kaggle dataset and copy its files into data/raw/."""
    import kagglehub  # dev dependency; only needed for this step

    config = load_config()
    source = config["source"]
    cache_dir = Path(kagglehub.dataset_download(source["kaggle_dataset"]))

    raw_dir = path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for kaggle_name, file_key in source["kaggle_files"].items():
        src = cache_dir / kaggle_name
        if not src.exists():
            raise FileNotFoundError(f"{kaggle_name!r} not found in the Kaggle download.")
        dst = raw_dir / config["files"][file_key]
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


if __name__ == "__main__":
    for file in download_raw():
        print(f"saved: {file}")
