"""Download the raw dataset into data/raw/.

Run `python -m apex.data.download` (or `make data-download`). Two sources,
chosen by `config → source`:

- `kaggle_dataset`: a Kaggle dataset, via kagglehub (the lead data)
- `url`: a zip file on the web, which may contain more zips (UCI Bank Marketing)

In both cases `source.files` maps the publisher's file names to the names in
`config → files`, so the rest of the code never depends on how they were named.
"""

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from apex.config import load_config, path


def find_in_zip(archive: zipfile.ZipFile, name: str) -> bytes | None:
    """The bytes of file `name` anywhere in the zip, also inside nested zips."""
    for member in archive.namelist():
        if Path(member).name == name and not member.startswith("__MACOSX"):
            return archive.read(member)
    for member in archive.namelist():
        if member.endswith(".zip"):
            found = find_in_zip(zipfile.ZipFile(io.BytesIO(archive.read(member))), name)
            if found is not None:
                return found
    return None


def download_raw() -> list[Path]:
    """Download the dataset and save its files into data/raw/ under the configured names."""
    config = load_config()
    source = config["source"]
    raw_dir = path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if "kaggle_dataset" in source:
        import kagglehub  # dev dependency; only needed for this step

        cache_dir = Path(kagglehub.dataset_download(source["kaggle_dataset"]))
    else:
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))

    copied = []
    for published_name, file_key in source["files"].items():
        dst = raw_dir / config["files"][file_key]
        if "kaggle_dataset" in source:
            src = cache_dir / published_name
            if not src.exists():
                raise FileNotFoundError(f"{published_name!r} not found in the Kaggle download.")
            shutil.copy2(src, dst)
        else:
            content = find_in_zip(archive, published_name)
            if content is None:
                raise FileNotFoundError(f"{published_name!r} not found in {source['url']}.")
            dst.write_bytes(content)
        copied.append(dst)
    return copied


if __name__ == "__main__":
    for file in download_raw():
        print(f"saved: {file}")
