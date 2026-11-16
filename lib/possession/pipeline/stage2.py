"""Stage 2: who has the ball inside each live block of stage 1.

Stage 1's dead frames are fixed. The live blocks (maximal runs of non-D graded frames) are filled with
H / A only, by the explicit-duration Viterbi decoder fed with the team model's P(home | alive).

Change cost: a change of team must be paid for by `change_cost_seconds` of clear evidence for the other
team ("clear" = odds of `clear_evidence_odds` per frame). A run of the other team inside a possession
costs two changes, so the penalty per change is seconds * 25 * log(odds) / 2 nats. No possession-count
target is used in stage 2.
"""

from dataclasses import replace

import numpy as np

from .. import timeline
from ..config import Config, Stage2Config
from ..constants import A, D, FPS, H
from ..decoding import viterbi
from ..decoding.layered import decode_with_layers
from .stage1 import ObservedLayers


def structured_proba(p_home: np.ndarray, dead: np.ndarray) -> np.ndarray:
    """(n, 3) H / A / D probabilities: D certain on dead frames and impossible on live ones.

    Parameters
    ----------
    p_home : numpy.ndarray
        (n,) the team model's P(H | alive); clipped to [0.001, 0.999].
    dead : numpy.ndarray
        (n,) bool, stage 1's D.

    Returns
    -------
    numpy.ndarray
        (n, 3) float64; P(D) is 0.998 on dead frames and 0.001 elsewhere.
    """
    p_home = np.clip(np.asarray(p_home, dtype=np.float64), 0.001, 0.999)
    p_dead = np.where(dead, 0.998, 0.001)
    return np.column_stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead])


def change_penalty(cfg: Stage2Config) -> float:
    """Switch penalty (nats per change of team) from the change cost in seconds of clear evidence.

    ``change_cost_seconds * FPS * log(clear_evidence_odds) / 2``: a run of the other team inside a
    possession costs two changes.

    Parameters
    ----------
    cfg : Stage2Config
        Uses `change_cost_seconds` and `clear_evidence_odds`.

    Returns
    -------
    float
        Nats, used as the decoder's ``penalty_scale``.
    """
    return cfg.change_cost_seconds * FPS * np.log(cfg.clear_evidence_odds) / 2


def p_home_given_alive(proba: np.ndarray) -> np.ndarray:
    """P(H | alive) from the team model's (n, 3) H / A / D probabilities.

    Parameters
    ----------
    proba : numpy.ndarray
        (n, 3) P(H), P(A), P(D).

    Returns
    -------
    numpy.ndarray
        (n,) float64 P(H) / (P(H) + P(A)).
    """
    proba = np.asarray(proba, dtype=np.float64)
    return proba[:, H] / np.maximum(proba[:, H] + proba[:, A], 1e-9)


def run_stage2(team_proba: np.ndarray, stage1: np.ndarray, layers: ObservedLayers, cfg: Config) -> np.ndarray:
    """Final H / A / D labels (must run inside timeline.use(challenge.timeline)).

    Stage 1's D frames are the hard layer; the team model's P(H | alive) is decoded with
    ``penalty_scale`` from `change_penalty` and no calibration.

    Parameters
    ----------
    team_proba : numpy.ndarray
        (n, 3) the team model's P(H), P(A), P(D).
    stage1 : numpy.ndarray
        (n,) stage-1 D / H (alive) / X codes.
    layers : ObservedLayers
        The forced layers (``stage1.observed_layers``).
    cfg : Config
        The pipeline configuration.

    Returns
    -------
    numpy.ndarray
        (n,) int8 H / A / D codes (every frame outside play is D).
    """
    s2 = cfg.stage2
    dead = (stage1 == D) & timeline.graded_mask(len(stage1))
    proba = structured_proba(p_home_given_alive(team_proba), dead)
    config = replace(viterbi.ViterbiConfig(**vars(cfg.viterbi)), penalty_scale=change_penalty(s2))
    labels, _ = decode_with_layers(proba, config, hard_dead=dead, certain_dead=layers.observed_dead,
                                   certain_alive=layers.observed_alive, overshoot_frames=s2.overshoot_frames,
                                   sliver_frames=s2.sliver_frames)
    return labels
