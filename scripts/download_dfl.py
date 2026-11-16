"""Download the DFL open data (Bassek et al. 2025, CC BY 4.0) into paths.dfl_data (data/external/dfl/).

    python scripts/download_dfl.py [--config configs/default.toml]

"An integrated dataset of spatiotemporal and event data in elite soccer", figshare article 28196177
(doi:10.6084/m9.figshare.28196177): 7 Bundesliga matches (2022/23), per match the match information,
the raw events and the 25 Hz positions (~2.6 GB in all). Every file is checked against figshare's size
and MD5 and downloaded again until it matches. Only the match information and positions files are read
by the pipeline; the events files are downloaded with the article but never used.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
RETRIES = 5


def md5(path: Path) -> str:
    """Hex MD5 of a file, read in 4 MB chunks."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, path: Path) -> None:
    """Download `url` to `path` through a ``.part`` file, renamed when complete."""
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 22):
            handle.write(chunk)
    partial.replace(path)


def main() -> None:
    """Command-line entry point: download every file of the article, up to RETRIES attempts each."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    from possession.config import DEFAULT_CONFIG, load_config
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    out = load_config(parser.parse_args().config).paths.dfl_data
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE, timeout=60) as response:
        files = json.load(response)["files"]
    for entry in files:
        path = out / entry["name"]
        attempt = 0
        while not (path.exists() and path.stat().st_size == entry["size"] and md5(path) == entry["computed_md5"]):
            attempt += 1
            if attempt > RETRIES:
                sys.exit(f"could not download {path.name}")
            print(f"get   {path.name} ({entry['size'] / 1e6:.0f} MB, attempt {attempt})", flush=True)
            try:
                fetch(entry["download_url"], path)
            except OSError as error:
                print(f"      failed: {error}", flush=True)
                time.sleep(5 * attempt)
        print(f"ok    {path.name}", flush=True)
    print(f"DFL open data complete in {out}")


if __name__ == "__main__":
    main()
