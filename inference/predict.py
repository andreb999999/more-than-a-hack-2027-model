"""Label every frame of the match with the trained models and write the submission CSV.

    python inference/predict.py [--config configs/default.toml]

Needs artifacts/ from train/train.py. Writes outputs/submission.csv (frame,label; 199,319 rows) and
outputs/model_outputs.npz, then runs the official format check (evaluation/validate_submission.py).
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))   # runs without `pip install -e .`

from possession.config import REPO_ROOT  # noqa: E402
from possession.pipeline.inference import predict, save_prediction  # noqa: E402
from possession.runtime import session  # noqa: E402


def main() -> None:
    """Command-line entry point: predict, write the CSV, exit with the format check's return code."""
    with session("Predict H / A / D for every frame and write the submission.") as cfg:
        save_prediction(predict(cfg), cfg.paths.submission)
    check = subprocess.run([sys.executable, str(REPO_ROOT / "evaluation" / "validate_submission.py"),
                            str(cfg.paths.submission)])
    sys.exit(check.returncode)


if __name__ == "__main__":
    main()
