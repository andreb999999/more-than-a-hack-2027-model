"""End-to-end reproduction: fetch the training data, check the inputs, train, predict, compare.

    python scripts/reproduce.py [--config configs/default.toml] [--skip-train] [--skip-download]

1. scripts/download_dfl.py: the DFL open data (skipped for files already present and MD5-verified);
2. scripts/check_data.py: every input against the checksums the submission was made from;
3. train/train.py (unless --skip-train) and inference/predict.py;
4. the SHA-256 of the produced CSV against SUBMITTED_SHA256 (the file that was submitted).

Only the organisers' match bundle has to be in place beforehand (docs/DATA.md). Exit code 1 if a file
differs.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMITTED_SHA256 = "46c69c3e559135a0e716eb316b67f13c953f1fc18378b3bace57fda242c21716"


def run(*args: str) -> None:
    """Run ``python <args>`` in the repository root; raises CalledProcessError on failure."""
    print("\n$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def main() -> None:
    """Command-line entry point: run the whole chain and compare the SHA-256."""
    sys.path.insert(0, str(REPO / "lib"))
    from possession.config import DEFAULT_CONFIG, load_config
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-train", action="store_true", help="reuse the models in artifacts/")
    parser.add_argument("--skip-download", action="store_true", help="do not fetch the DFL open data")
    args = parser.parse_args()
    config = str(args.config.resolve())
    if not args.skip_download:
        run("scripts/download_dfl.py", "--config", config)
    run("scripts/check_data.py", "--config", config)
    if not args.skip_train:
        run("train/train.py", "--config", config)
    run("inference/predict.py", "--config", config)
    submission = load_config(args.config).paths.submission
    sha = hashlib.sha256(submission.read_bytes()).hexdigest()
    print(f"\n{submission}\n  sha256    {sha}\n  submitted {SUBMITTED_SHA256}")
    if sha != SUBMITTED_SHA256:
        print("DIFFERENT from the submitted file")
        sys.exit(1)
    print("IDENTICAL to the submitted file")


if __name__ == "__main__":
    main()
