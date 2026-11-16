"""Train the dead/alive head and the team model on the DFL open data.

    python train/train.py [--config configs/default.toml]

Reads the DFL matches (scripts/download_dfl.py) and the challenge match's unlabelled tracking (for the sensor
profiles), writes artifacts/deadball_head.joblib, artifacts/team_model.joblib and
artifacts/training_manifest.json. First run ~30 min (DFL parsing and degradation are cached in cache/).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))   # runs without `pip install -e .`

from possession.pipeline.training import train  # noqa: E402
from possession.runtime import session  # noqa: E402


def main() -> None:
    """Command-line entry point: load the configuration and train."""
    with session("Train the DFL dead/alive head and team model.") as cfg:
        train(cfg)


if __name__ == "__main__":
    main()
