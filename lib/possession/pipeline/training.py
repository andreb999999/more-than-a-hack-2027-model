"""Training: fit the dead/alive head and the team model on the DFL matches, save them as artifacts.

The challenge match enters training only through its unlabelled tracking: the sensor profiles that the DFL
tracking is degraded to (training_data.degradation_profiles). No label of the challenge match is read.
"""

import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn

from ..config import Config
from ..models.deadball_head import fit_head
from ..models.team_model import TeamModel
from . import training_data
from .challenge_match import load_challenge_match

log = logging.getLogger(__name__)

HEAD_FILE, TEAM_FILE, MANIFEST_FILE = "deadball_head.joblib", "team_model.joblib", "training_manifest.json"


@dataclass
class TrainedModels:
    """The two learned components.

    Attributes
    ----------
    head : sklearn.ensemble.HistGradientBoostingClassifier
        The dead/alive head.
    team : TeamModel
        The team model.
    """
    head: object          # sklearn HistGradientBoostingClassifier
    team: TeamModel


def train(cfg: Config) -> TrainedModels:
    """Fit both models on the DFL matches and save them to ``paths.artifacts``.

    The challenge match contributes only its unlabelled tracking (the degradation profiles). The team model is
    restricted to the DFL feature columns that also exist on the challenge match.

    Parameters
    ----------
    cfg : Config
        The pipeline configuration.

    Returns
    -------
    TrainedModels
        The fitted models (also written, with a JSON training manifest).
    """
    challenge = load_challenge_match(cfg)
    speed_profile, state_profile = training_data.degradation_profiles(cfg, challenge)
    log.info("degradation profiles: ball missing %.3f in play, %.3f in stoppages",
             state_profile.play_missing_share, state_profile.stop_missing_share)

    rows = training_data.head_rows(cfg, state_profile)
    head = fit_head(rows, cfg.stage1.head, cfg.runtime.seed)
    log.info("dead/alive head fitted on %d rows of %d DFL matches", sum(len(r[1]) for r in rows), len(rows))

    features, labels = training_data.team_table(cfg, speed_profile)
    columns = [c for c in features.columns if c in challenge.features.columns]
    team = TeamModel(cfg.team_model, cfg.runtime.seed).fit(features[columns], labels)
    log.info("team model fitted on %d rows x %d shape columns", len(labels), len(team.columns_))

    save(TrainedModels(head=head, team=team), cfg.paths.artifacts, {
        "dfl_matches": list(cfg.dfl.matches),
        "head_rows": sum(len(r[1]) for r in rows),
        "head_dead_share": float(np.concatenate([r[1] for r in rows]).mean()),
        "team_rows": int(len(labels)),
        "team_label_counts": {k: int(v) for k, v in zip("HAD", np.bincount(labels, minlength=3))},
        "team_columns": team.columns_,
        "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
    })
    return TrainedModels(head=head, team=team)


def save(models: TrainedModels, folder: Path, manifest: dict) -> None:
    """Write the models (joblib) and the manifest (JSON) to `folder`.

    Parameters
    ----------
    models : TrainedModels
        The fitted models.
    folder : Path
        Artifacts folder; created if needed.
    manifest : dict
        JSON-serialisable training summary, written as ``training_manifest.json``.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.head, folder / HEAD_FILE)
    joblib.dump(models.team, folder / TEAM_FILE)
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    log.info("models saved to %s", folder)


def load(folder: Path) -> TrainedModels:
    """Read the models written by `save`.

    Parameters
    ----------
    folder : Path
        Artifacts folder.

    Returns
    -------
    TrainedModels

    Raises
    ------
    FileNotFoundError
        If a model file is missing (train first).
    """
    folder = Path(folder)
    missing = [f for f in (HEAD_FILE, TEAM_FILE) if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(f"no trained models in {folder} ({missing}): run python train/train.py first")
    return TrainedModels(head=joblib.load(folder / HEAD_FILE), team=joblib.load(folder / TEAM_FILE))
