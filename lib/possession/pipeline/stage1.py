"""Stage 1: dead ball vs alive on every frame of the challenge match.

    P(dead) = head_weight * (DFL-trained dead/alive head) + (1 - head_weight) * (hand-weighted cue rules)

decoded by the explicit-duration Viterbi decoder with three forced layers (decoding/layered.py):
    hard dead     rules.forced: outside play, ball out until the restart, goal to kick-off, set pieces
    observed dead rules.observed.certain_dead (+ outside play): forced again after the playbook
    certainly live rules.observed.restart_alive: no D emission
and calibrated to two label-free priors (44 % dead time, at most 340 live stretches) with a neutral team split
(only dead vs alive matters here). After the decode, L5 (rules.observed.keeper_holds_untracked): stoppages
where the goalkeeper holds an untracked ball become alive, and alive slivers of at most 1 s left between
two D runs become D again.
"""

import logging
from dataclasses import dataclass

import numpy as np

from .. import timeline
from ..config import Config
from ..constants import D, H, X
from ..decoding import viterbi
from ..decoding.layered import Calibration, compose, decode_with_layers
from ..features.deadball import cue_rule_probability, rule_terms
from ..rules import forced, observed
from .challenge_match import ChallengeMatch

log = logging.getLogger(__name__)


@dataclass
class ObservedLayers:
    """The forced layers of the challenge match, shared by both stages.

    Attributes
    ----------
    hard_dead : numpy.ndarray
        (n,) bool forced dead-ball intervals (``rules.forced.forced_dead_mask``), graded frames only.
    observed_dead : numpy.ndarray
        (n,) bool directly observed D (``observed.certain_dead``) plus every frame outside play.
    observed_alive : numpy.ndarray
        (n,) bool certainly live frames (``observed.restart_alive``).
    """
    hard_dead: np.ndarray
    observed_dead: np.ndarray
    observed_alive: np.ndarray


def observed_layers(challenge: ChallengeMatch) -> ObservedLayers:
    """The forced layers of the challenge match (must run inside timeline.use(challenge.timeline)).

    Parameters
    ----------
    challenge : ChallengeMatch
        The featurised challenge match.

    Returns
    -------
    ObservedLayers
    """
    trk, tags, speed = challenge.tracking, challenge.tags, challenge.player_speed
    return ObservedLayers(
        hard_dead=forced.forced_dead_mask(trk, tags, speed, challenge.sp_open),
        observed_dead=observed.certain_dead(trk) | observed.pre_kickoff_mask(trk, tags),
        observed_alive=observed.restart_alive(trk, player_speed=speed))


def cue_rules_dead(challenge: ChallengeMatch, cfg: Config) -> np.ndarray:
    """(n,) P(dead) of the hand-weighted cue rules.

    Parameters
    ----------
    challenge : ChallengeMatch
        The featurised challenge match (its dead-ball cues are used).
    cfg : Config
        Uses ``stage1.cue_rules`` (bias and weights).

    Returns
    -------
    numpy.ndarray
        (n,) float64 P(dead).

    Raises
    ------
    ValueError
        If the configured weights do not name exactly the cue terms of ``features.deadball.rule_terms``.
    """
    rules = cfg.stage1.cue_rules
    terms = rule_terms(challenge.cues)
    if set(rules.weights) != set(terms.columns):
        raise ValueError(f"[stage1.cue_rules.weights] must name exactly the cue terms {list(terms.columns)}")
    return cue_rule_probability(terms, rules.weights, rules.bias)


def apply_keeper_holds(stage1: np.ndarray, challenge: ChallengeMatch, sliver_frames: int) -> np.ndarray:
    """L5 after the decode: D frames where the goalkeeper holds an untracked ball become alive; alive
    slivers shorter than `sliver_frames` left between D are D again. Returns D / H(alive) / X codes.

    Parameters
    ----------
    stage1 : numpy.ndarray
        (n,) decoded stage-1 label codes.
    challenge : ChallengeMatch
        The featurised challenge match.
    sliver_frames : int
        Alive runs shorter than this many frames between two D runs become D.

    Returns
    -------
    numpy.ndarray
        (n,) int8: D on dead graded frames, H on alive graded frames, X on ungraded frames.
    """
    graded = timeline.graded_mask(len(stage1))
    dead = (stage1 == D) & graded & ~np.asarray(observed.keeper_holds_untracked(challenge.tracking), dtype=bool)
    for start, stop in timeline.halves():
        seg = np.where(dead[start:stop], D, H).astype(np.int8)
        dead[start:stop] = viterbi.remove_sandwiched_alive(seg, sliver_frames) == D
    return np.where(dead, D, np.where(graded, H, X)).astype(np.int8)


def run_stage1(head_dead: np.ndarray, challenge: ChallengeMatch, layers: ObservedLayers, cfg: Config) -> np.ndarray:
    """Stage-1 labels: D on dead frames, H on alive graded frames (team not decided), X elsewhere.

    P(dead) mixes the head and the cue rules by ``stage1.head_weight`` and is decoded by `decode_dead`.
    Must run inside ``timeline.use(challenge.timeline)``.

    Parameters
    ----------
    head_dead : numpy.ndarray
        (n,) the DFL head's P(dead) per frame.
    challenge : ChallengeMatch
        The featurised challenge match.
    layers : ObservedLayers
        The forced layers (`observed_layers`).
    cfg : Config
        The pipeline configuration.

    Returns
    -------
    numpy.ndarray
        (n,) int8 D / H (alive) / X codes.
    """
    weight = cfg.stage1.head_weight
    return decode_dead(weight * head_dead + (1 - weight) * cue_rules_dead(challenge, cfg), challenge, layers, cfg)


def decode_dead(p_dead: np.ndarray, challenge: ChallengeMatch, layers: ObservedLayers, cfg: Config) -> np.ndarray:
    """Decode a per-frame P(dead) into D / alive with the forced layers and stage 1's calibration.

    The decode uses a neutral team split (P(H | alive) = 0.5) through `decode_with_layers`, then
    `apply_keeper_holds` (L5). Must run inside ``timeline.use(challenge.timeline)``.

    Parameters
    ----------
    p_dead : numpy.ndarray
        (n,) P(dead) per frame.
    challenge : ChallengeMatch
        The featurised challenge match.
    layers : ObservedLayers
        The forced layers (`observed_layers`).
    cfg : Config
        The pipeline configuration.

    Returns
    -------
    numpy.ndarray
        (n,) int8 D / H (alive) / X codes.
    """
    s1 = cfg.stage1
    proba = compose(np.full(len(p_dead), 0.5), p_dead)
    labels, decoder = decode_with_layers(proba, viterbi.ViterbiConfig(**vars(cfg.viterbi)), hard_dead=layers.hard_dead,
                                   certain_dead=layers.observed_dead, certain_alive=layers.observed_alive,
                                   overshoot_frames=s1.overshoot_frames, sliver_frames=s1.sliver_frames,
                                   calibration=Calibration(**vars(s1.calibration)))
    log.info("stage 1 decoder calibrated: penalty_scale %.4f, dead_bias %.4f", decoder.penalty_scale, decoder.dead_bias)
    return apply_keeper_holds(labels, challenge, s1.keeper_hold_sliver_frames)
