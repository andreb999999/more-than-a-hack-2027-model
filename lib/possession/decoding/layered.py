"""Decoding with forced layers: the one decode used by both stages.

    hard_dead      frames that must be D whatever the model says (forced rules, or stage 1's D in stage 2).
                   They get a near-certain D emission before decoding, so the decoder places its boundaries
                   around them, and are forced to D after decoding; alive blips left between forced and
                   decoded D are removed by the sandwich rule.
    certain_dead   directly observed D (rules.observed.certain_dead). Treated as hard_dead and, in addition,
                   forced to D again after the playbook, so its 1 s overshoot never trims a stoppage that
                   started exactly when the ball crossed the line; alive slivers of at most 1 s left between
                   it and a stoppage are then D (`sliver_frames`).
    certain_alive  certainly live frames (rules.observed.restart_alive): their D emission is set to ~0 (team
                   split kept) unless the frame is also hard D.

Calibration (optional) chooses dead_bias and penalty_scale from two priors, no ground truth: the D-share
prior describes the model's own uncertain dead-ball evidence, so it is calibrated without the hard layer;
the live-stretch prior describes the final output, so it is calibrated with it.
"""

from dataclasses import dataclass

import numpy as np

from .. import timeline
from ..constants import D, X
from . import viterbi
from .playbook import apply_playbook

FORCED_DEAD_EMISSION = (0.001, 0.001, 0.998)
ALIVE_TEAM_MASS, ALIVE_DEAD_MASS = 0.999, 0.001


@dataclass(frozen=True)
class Calibration:
    """label-free calibration targets of the decoder and the bisection settings that reach them.

    Same fields as ``config.CalibrationConfig``.

    Attributes
    ----------
    target_live_stretches : int
        Target number of live stretches (H / A runs; calibrated with the hard layer).
    target_dead_share : float or None
        Target D share of match play (calibrated without the hard layer); None skips that calibration.
    rounds : int
        Number of (dead share, live stretches) rounds.
    penalty_range : tuple of float
        (low, high) geometric bisection interval of ``penalty_scale``.
    penalty_iterations : int
        Maximum bisection steps for ``penalty_scale``.
    live_stretch_tolerance : float
        Stop when the number of live stretches is this close to the target.
    dead_bias_range : tuple of float
        (low, high) linear bisection interval of ``dead_bias``, nats.
    dead_bias_iterations : int
        Maximum bisection steps for ``dead_bias``.
    dead_share_tolerance : float
        Stop when the dead share is this close to the target.
    """
    target_live_stretches: int
    target_dead_share: float | None
    rounds: int
    penalty_range: tuple[float, float]
    penalty_iterations: int
    live_stretch_tolerance: float
    dead_bias_range: tuple[float, float]
    dead_bias_iterations: int
    dead_share_tolerance: float


def decode_with_layers(proba: np.ndarray, config: viterbi.ViterbiConfig, hard_dead: np.ndarray | None,
                       certain_dead: np.ndarray, certain_alive: np.ndarray, overshoot_frames: int,
                       sliver_frames: int, calibration: Calibration | None = None
                       ) -> tuple[np.ndarray, viterbi.ViterbiConfig]:
    """(labels, the decoder configuration used) for (n, 3) H / A / D probabilities.

    Steps: certainly-live frames get D mass ALIVE_DEAD_MASS (team split kept); hard-dead frames get
    FORCED_DEAD_EMISSION; optional calibration; decode; hard-dead frames forced to D and the sandwich
    rule (``config.max_sandwiched_alive``); the playbook (free frames, overshoot); certain-dead frames
    forced to D again; then alive slivers shorter than `sliver_frames` between D runs become D. Must run
    inside a timeline with known kick-offs.

    Parameters
    ----------
    proba : numpy.ndarray
        (n, 3) H / A / D probabilities.
    config : viterbi.ViterbiConfig
        Decoder parameters (the calibration starts from them).
    hard_dead : numpy.ndarray or None
        (n,) bool frames that must be D; None uses `certain_dead` alone (it is always included).
    certain_dead : numpy.ndarray
        (n,) bool directly observed D.
    certain_alive : numpy.ndarray
        (n,) bool certainly live frames.
    overshoot_frames : int
        Frames by which possessions are extended into each neighbouring stoppage (0: off).
    sliver_frames : int
        Alive runs shorter than this many frames between two D runs become D after the playbook.
    calibration : Calibration, optional
        If given, choose `dead_bias` and `penalty_scale` from the label-free priors.

    Returns
    -------
    labels : numpy.ndarray
        (n,) int8 label codes H / A / D; the playbook makes every frame outside play D (before each
        kick-off and after each half, ungraded frames included).
    config : viterbi.ViterbiConfig
        The decoder configuration used (calibrated if `calibration` was given).
    """
    hard_dead = certain_dead if hard_dead is None else (hard_dead | certain_dead)
    proba = np.asarray(proba, dtype=np.float64)
    live = certain_alive & ~hard_dead
    proba = proba.copy()
    team = proba[live, :D]
    team = team / np.maximum(team.sum(1, keepdims=True), 1e-9)
    proba[live, :D] = ALIVE_TEAM_MASS * team
    proba[live, D] = ALIVE_DEAD_MASS
    raw = proba
    proba = proba.copy()
    proba[hard_dead] = FORCED_DEAD_EMISSION
    if calibration is not None:
        for _ in range(calibration.rounds):          # the two calibrations interact only weakly
            if calibration.target_dead_share is not None:
                config = viterbi.calibrate_dead_bias(raw, config, calibration.target_dead_share,
                                                     *calibration.dead_bias_range, calibration.dead_bias_iterations,
                                                     calibration.dead_share_tolerance)
            config = viterbi.calibrate_penalty_scale(proba, config, calibration.target_live_stretches,
                                                     *calibration.penalty_range, calibration.penalty_iterations,
                                                     calibration.live_stretch_tolerance)
    decoded = viterbi.decode(proba, config).copy()
    decoded[hard_dead & (decoded != X)] = D
    for start, stop in timeline.halves():
        decoded[start:stop] = viterbi.remove_sandwiched_alive(decoded[start:stop], config.max_sandwiched_alive)
    labels = apply_playbook(decoded, overshoot_frames).copy()
    labels[certain_dead & (labels != X)] = D
    for start, stop in timeline.halves():
        labels[start:stop] = viterbi.remove_sandwiched_alive(labels[start:stop], sliver_frames)
    return labels, config


def compose(p_home_given_alive: np.ndarray, p_dead: np.ndarray) -> np.ndarray:
    """(n, 3) H / A / D probabilities from a team stream and a dead-ball stream.

    Parameters
    ----------
    p_home_given_alive : numpy.ndarray
        (n,) P(H | alive), clipped to [0, 1].
    p_dead : numpy.ndarray
        (n,) P(D), clipped to [0, 1].

    Returns
    -------
    numpy.ndarray
        (n, 3) float64 columns ``(1 - p_dead) * p_home``, ``(1 - p_dead) * (1 - p_home)``, ``p_dead``.
    """
    p_home = np.clip(np.asarray(p_home_given_alive, dtype=np.float64), 0.0, 1.0)
    p_dead = np.clip(np.asarray(p_dead, dtype=np.float64), 0.0, 1.0)
    return np.column_stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead])
