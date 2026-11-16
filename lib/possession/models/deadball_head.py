"""The dead/alive head: a gradient-boosted classifier of P(dead), trained on the 7 DFL matches.

Input per frame: the shape view of the team features plus the tag-free dead-ball cues, and the cues
1 s and 2 s ahead (the scene after a whistle tells whether it was a whistle), restricted to the 40 columns
of config `stage1.head.columns` (10 feature groups). Target: DFL BallStatus (dead = 1).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import DeadballHeadConfig
from ..constants import FPS


@dataclass
class HeadTable:
    """What the head needs from one match: shape features, tag-free cues, and (DFL only) the labels.

    Attributes
    ----------
    shape : pandas.DataFrame
        (n, n_shape) float32 shape view of the team features, index 0..n-1.
    cues : pandas.DataFrame
        (n, n_cues) float32 tag-free dead-ball cues (``features.deadball.tag_free_cues``).
    dead : numpy.ndarray or None
        (n,) bool, DFL BallStatus dead; None on the challenge match.
    judged : numpy.ndarray or None
        (n,) bool, labelled graded frames; None on the challenge match.
    """
    shape: pd.DataFrame
    cues: pd.DataFrame
    dead: np.ndarray | None = None      # (n,) bool, DFL BallStatus dead
    judged: np.ndarray | None = None    # (n,) bool, labelled graded frames


def head_features(table: HeadTable, columns: list[str], lookahead_s: tuple[int, ...]) -> pd.DataFrame:
    """shape + cues + cues `lookahead_s` seconds ahead, restricted to `columns` (in that order).

    Parameters
    ----------
    table : HeadTable
        One match.
    columns : list of str
        Columns to keep, in order; look-ahead copies are named ``<cue>__ahead<s>s``.
    lookahead_s : tuple of int
        Look-ahead offsets in seconds; the last frames of a match get NaN look-ahead values.

    Returns
    -------
    pandas.DataFrame
        (n, len(columns)).
    """
    cues = table.cues
    parts = [table.shape, cues]
    for s in lookahead_s:
        ahead = cues.shift(-s * FPS)
        ahead.columns = [f"{c}__ahead{s}s" for c in cues.columns]
        parts.append(ahead)
    return pd.concat(parts, axis=1)[list(columns)]


def training_rows(table: HeadTable, cfg: DeadballHeadConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """(x, y) of one DFL match: its judged frames, one in `cfg.row_stride` (neighbouring frames are
    near-copies). The look-ahead columns are computed on the full table first.

    Parameters
    ----------
    table : HeadTable
        One DFL match, with `dead` and `judged` set.
    cfg : DeadballHeadConfig
        Columns, look-ahead and row stride.

    Returns
    -------
    x : pandas.DataFrame
        (n_rows, len(cfg.columns)) head input.
    y : numpy.ndarray
        (n_rows,) bool, dead.
    """
    x = head_features(table, cfg.columns, cfg.lookahead_s).loc[table.judged].iloc[::cfg.row_stride]
    return x, table.dead[table.judged][::cfg.row_stride]


def fit_head(rows: list[tuple[pd.DataFrame, np.ndarray]], cfg: DeadballHeadConfig, seed: int) -> HistGradientBoostingClassifier:
    """Fit the head on the training rows of the DFL matches (training_rows).

    Parameters
    ----------
    rows : list of tuple
        ``(x, y)`` per DFL match, from `training_rows`; concatenated in order.
    cfg : DeadballHeadConfig
        Gradient-boosting hyperparameters.
    seed : int
        ``random_state`` of the classifier (no early stopping).

    Returns
    -------
    sklearn.ensemble.HistGradientBoostingClassifier
        The fitted head; class 1 is dead.
    """
    x = pd.concat([r[0] for r in rows], ignore_index=True)
    y = np.concatenate([r[1] for r in rows]).astype(int)
    model = HistGradientBoostingClassifier(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
                                           max_leaf_nodes=cfg.max_leaf_nodes, l2_regularization=cfg.l2_regularization,
                                           random_state=seed, early_stopping=False)
    return model.fit(x, y)


def predict_dead(model: HistGradientBoostingClassifier, table: HeadTable, cfg: DeadballHeadConfig) -> np.ndarray:
    """(n,) P(dead) of the head on one match.

    Parameters
    ----------
    model : sklearn.ensemble.HistGradientBoostingClassifier
        The fitted head (`fit_head`).
    table : HeadTable
        The match to score (labels not needed).
    cfg : DeadballHeadConfig
        Columns and look-ahead, as used for training.

    Returns
    -------
    numpy.ndarray
        (n,) float64 P(dead).
    """
    return model.predict_proba(head_features(table, cfg.columns, cfg.lookahead_s))[:, 1]
