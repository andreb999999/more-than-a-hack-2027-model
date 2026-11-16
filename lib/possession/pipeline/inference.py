"""Inference: the challenge match + the trained models -> the submission CSV.

    1. featurise the challenge match (team features, dead-ball cues), detect the kick-offs
    2. model outputs: the head's P(dead) and the team model's P(H), P(A), P(D) per frame
    3. stage 1: dead vs alive (pipeline/stage1.py)
    4. stage 2: home vs away on the live frames (pipeline/stage2.py)
    5. write and validate the CSV

Model outputs are stored as float32 (outputs/model_outputs.npz) and the decoders read the stored values.
The decoders compare log-probabilities near decision edges, so this keeps the CSV reproducible to the frame.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import timeline
from ..config import Config
from ..constants import A, D, H
from ..io.submission import write_submission
from ..models.deadball_head import predict_dead
from . import training
from .challenge_match import load_challenge_match
from .stage1 import observed_layers, run_stage1
from .stage2 import run_stage2

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    """The outputs of one inference run.

    Attributes
    ----------
    labels : numpy.ndarray
        (n,) final label codes, written to the CSV; the decoder's playbook makes every frame outside
        play D (ungraded frames included), so in practice no X is produced.
    stage1 : numpy.ndarray
        (n,) stage-1 D / alive (H) / X codes.
    head_dead : numpy.ndarray
        (n,) stored (float32-rounded) P(dead) of the head, as float64.
    team_proba : numpy.ndarray
        (n, 3) stored P(H), P(A), P(D) of the team model, as float64.
    """
    labels: np.ndarray        # (n,) final H / A / D codes (D outside play)
    stage1: np.ndarray        # (n,) D / alive (H) / X
    head_dead: np.ndarray     # (n,) stored P(dead) of the head
    team_proba: np.ndarray    # (n, 3) stored P(H), P(A), P(D) of the team model


def stored(values: np.ndarray) -> np.ndarray:
    """Model outputs as stored (float32) and read back for decoding (float64).

    Parameters
    ----------
    values : numpy.ndarray
        Model outputs.

    Returns
    -------
    numpy.ndarray
        float64 values rounded through float32.
    """
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def predict(cfg: Config) -> Prediction:
    """Run the full inference on the challenge match with the trained models.

    Parameters
    ----------
    cfg : Config
        The pipeline configuration (models are read from ``paths.artifacts``).

    Returns
    -------
    Prediction

    Raises
    ------
    FileNotFoundError
        If the trained models or the match data are missing.
    """
    challenge = load_challenge_match(cfg)
    models = training.load(cfg.paths.artifacts)
    head_dead = stored(predict_dead(models.head, challenge.head_table(), cfg.stage1.head))
    team_proba = stored(models.team.predict_proba(challenge.features))
    with timeline.use(challenge.timeline):
        layers = observed_layers(challenge)
        stage1 = run_stage1(head_dead, challenge, layers, cfg)
        labels = run_stage2(team_proba, stage1, layers, cfg)
    return Prediction(labels=labels, stage1=stage1, head_dead=head_dead, team_proba=team_proba)


def save_prediction(prediction: Prediction, submission: Path) -> None:
    """The submission CSV, and next to it the stored model outputs and stage 1 (for inspection).

    Parameters
    ----------
    prediction : Prediction
        Output of `predict`.
    submission : Path
        The CSV path; ``model_outputs.npz`` is written in the same folder.
    """
    write_submission(prediction.labels, submission)
    np.savez_compressed(Path(submission).with_name("model_outputs.npz"),
                        head_dead=prediction.head_dead.astype(np.float32),
                        team_proba=prediction.team_proba.astype(np.float32), stage1=prediction.stage1)
    graded = timeline.graded_mask(len(prediction.labels))
    labels = prediction.labels[graded]
    log.info("wrote %s: graded frames H %.1f %%, A %.1f %%, D %.1f %%", submission,
             100 * (labels == H).mean(), 100 * (labels == A).mean(), 100 * (labels == D).mean())
