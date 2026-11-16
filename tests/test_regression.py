"""The produced submission, when present, is byte-identical to the submitted one."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from possession.config import load_config

REPO = Path(__file__).resolve().parents[1]
SUBMISSION = REPO / "outputs" / "submission.csv"


def _submitted_sha256() -> str:
    spec = importlib.util.spec_from_file_location("reproduce", REPO / "scripts" / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SUBMITTED_SHA256


def test_default_submission_path_is_outputs_submission_csv():
    assert load_config().paths.submission == SUBMISSION.resolve()


def test_submitted_sha256_is_a_sha256_digest():
    digest = _submitted_sha256()
    assert len(digest) == 64 and int(digest, 16) >= 0


@pytest.mark.skipif(not SUBMISSION.is_file(), reason="outputs/submission.csv not produced (run inference/predict.py)")
def test_submission_matches_the_submitted_sha256():
    assert hashlib.sha256(SUBMISSION.read_bytes()).hexdigest() == _submitted_sha256()
