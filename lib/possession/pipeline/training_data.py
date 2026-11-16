"""Training tables built from the 7 DFL matches, each degraded to look like the challenge tracking.

    degradation_profiles   the two sensor profiles of the challenge tracking (the speed-bin profile from the tracking alone;
                           the game-state profile from the tracking split by `stoppage_state`)
    head_rows              per DFL match, game-state-degraded: the head's input columns and BallStatus, one judged
                           frame in `stage1.head.row_stride`
    team_table             all DFL matches, speed-bin-degraded: team features and raw H / A / D labels, one
                           labelled frame in `dfl.row_stride`
"""

import logging
import sys

import numpy as np
import pandas as pd

from .. import cache, timeline
from ..config import Config
from ..constants import D, X
from ..degradation import profiles, simulate
from ..features import deadball, team
from ..io import dfl as dfl_io
from ..io.tracking import empty_tags
from ..models import deadball_head
from ..models.deadball_head import HeadTable
from ..rules import observed
from . import stage1
from .challenge_match import FEATURE_CODE, ChallengeMatch

log = logging.getLogger(__name__)


def stoppage_state(cfg: Config, challenge: ChallengeMatch) -> np.ndarray:
    """(n,) bool stoppages of the challenge match from its tracking and Smart Tagging, without any model.

    The hand-weighted cue rules (`stage1.cue_rules_dead`) decoded with the forced layers
    (`stage1.decode_dead`), restricted to match play. Must run inside ``timeline.use(challenge.timeline)``.

    Parameters
    ----------
    cfg : Config
        The pipeline configuration.
    challenge : ChallengeMatch
        The featurised challenge match.

    Returns
    -------
    numpy.ndarray
        (n,) bool, True on stoppage frames.
    """
    layers = stage1.observed_layers(challenge)
    labels = stage1.decode_dead(stage1.cue_rules_dead(challenge, cfg), challenge, layers, cfg)
    return (labels == D) & timeline.match_mask(challenge.tracking.n_frames)


def degradation_profiles(cfg: Config, challenge: ChallengeMatch) -> tuple[profiles.SpeedBinProfile, profiles.GameStateProfile]:
    """(speed-bin, game-state) sensor profiles of the challenge tracking, from the tracking alone.

    Parameters
    ----------
    cfg : Config
        The pipeline configuration (`stoppage_state`).
    challenge : ChallengeMatch
        The featurised challenge match.

    Returns
    -------
    speed_profile : profiles.SpeedBinProfile
    state_profile : profiles.GameStateProfile
    """
    with timeline.use(challenge.timeline):
        speed_profile = profiles.measure_speed_bin_profile(challenge.tracking)
        state_profile = profiles.measure_game_state_profile(challenge.tracking, stoppage_state(cfg, challenge), speed_profile)
    return speed_profile, state_profile


def _head_table(match: dfl_io.DflMatch, state_profile: profiles.GameStateProfile, seed: int, rules: observed.OutRules) -> HeadTable:
    """The head's input table of one game-state-degraded DFL match, with BallStatus dead and judged frames."""
    dead = match.labels == D
    degraded = simulate.degrade_with_stoppage_ball(match.tracking, state_profile, dead, seed)
    frame = team.team_features(degraded, match.sign)
    with timeline.use(timeline.Timeline(halves=match.halves, kickoffs=(match.halves[0][0], match.halves[1][0]))):
        cues = deadball.dead_ball_cues(degraded, empty_tags(), rules)
        judged = timeline.graded_mask(len(match.labels)) & (match.labels != X)
    return HeadTable(shape=frame[team.shape_columns(frame)].reset_index(drop=True).astype(np.float32),
                     cues=deadball.tag_free_cues(cues).astype(np.float32), dead=dead, judged=judged)


def head_rows(cfg: Config, state_profile: profiles.GameStateProfile) -> list[tuple[pd.DataFrame, np.ndarray]]:
    """The dead/alive head's training rows of each game-state-degraded DFL match (cached per match).

    Parameters
    ----------
    cfg : Config
        Uses ``dfl.matches``, ``paths``, ``runtime.seed``, ``stage1.head`` and ``stage1.cues``.
    state_profile : profiles.GameStateProfile
        The challenge tracking's game-state profile.

    Returns
    -------
    list of tuple
        ``(x, y)`` per DFL match in ``cfg.dfl.matches`` order (``deadball_head.training_rows``).
    """
    out = []
    for name in cfg.dfl.matches:
        key = cache.fingerprint(*FEATURE_CODE, sys.modules[__name__], profiles, simulate, dfl_io, deadball_head,
                                *dfl_io.source_files(cfg.paths.dfl_data, name), repr(state_profile), cfg.runtime.seed,
                                cfg.stage1.head, cfg.stage1.cues)

        def build() -> tuple[pd.DataFrame, np.ndarray]:
            """Head rows of this DFL match."""
            match = dfl_io.load_match(cfg.paths.dfl_data, name, cfg.paths.cache)
            rules = observed.OutRules(cfg.stage1.cues.flight_veto, cfg.stage1.cues.inward_throw)
            return deadball_head.training_rows(_head_table(match, state_profile, cfg.runtime.seed, rules), cfg.stage1.head)

        out.append(cache.cached(cfg.paths.cache, f"head_rows_{name}", key, build))
        log.info("DFL %s: %d dead/alive head rows", name, len(out[-1][1]))
    return out


def _team_rows(match: dfl_io.DflMatch, speed_profile: profiles.SpeedBinProfile, seed: int, stride: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Team features and labels of one speed-bin-degraded DFL match, one labelled frame in `stride`."""
    frame = team.team_features(simulate.degrade_by_speed_bin(match.tracking, speed_profile, seed), match.sign)
    keep = np.flatnonzero(match.labels != X)[::stride]
    return frame.iloc[keep].reset_index(drop=True), match.labels[keep].astype(np.int8)


def team_table(cfg: Config, speed_profile: profiles.SpeedBinProfile) -> tuple[pd.DataFrame, np.ndarray]:
    """(features, labels) of all DFL matches, speed-bin-degraded, one labelled frame in `dfl.row_stride` (cached).

    Parameters
    ----------
    cfg : Config
        Uses ``dfl.matches``, ``dfl.row_stride``, ``paths`` and ``runtime.seed``.
    speed_profile : profiles.SpeedBinProfile
        The challenge tracking's speed-bin profile.

    Returns
    -------
    features : pandas.DataFrame
        Team features of the kept frames, all matches concatenated.
    labels : numpy.ndarray
        (n_rows,) int8 H / A / D codes (X frames are not kept).
    """
    frames, labels = [], []
    for name in cfg.dfl.matches:
        key = cache.fingerprint(*FEATURE_CODE, sys.modules[__name__], profiles, simulate, dfl_io,
                                *dfl_io.source_files(cfg.paths.dfl_data, name), repr(speed_profile), cfg.runtime.seed,
                                cfg.dfl.row_stride)

        def build() -> tuple[pd.DataFrame, np.ndarray]:
            """Team-model rows of this DFL match."""
            match = dfl_io.load_match(cfg.paths.dfl_data, name, cfg.paths.cache)
            return _team_rows(match, speed_profile, cfg.runtime.seed, cfg.dfl.row_stride)

        frame, lab = cache.cached(cfg.paths.cache, f"team_rows_{name}", key, build)
        frames.append(frame)
        labels.append(lab)
        log.info("DFL %s: %d team-model rows", name, len(lab))
    return pd.concat(frames, ignore_index=True), np.concatenate(labels)
