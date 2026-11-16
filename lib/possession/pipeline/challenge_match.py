"""Everything computed once from the challenge match's raw data (tracking + Smart Tagging), no labels."""

import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import cache, constants, timeline
from ..config import Config
from ..constants import GRADED_HALVES
from ..features import common, deadball, team
from ..io import tracking as tracking_io
from ..io.tracking import Tracking
from ..models.deadball_head import HeadTable
from ..rules import formations, forced, kinematics, observed
from ..timeline import Timeline

log = logging.getLogger(__name__)

# modules whose source enters the cache key of every feature table
FEATURE_CODE = (constants, timeline, common, deadball, team, observed, formations, forced, kinematics, tracking_io)


@dataclass
class ChallengeMatch:
    """The challenge match, featurised (no labels).

    Attributes
    ----------
    tracking : Tracking
        The tactical ATD in metres.
    tags : pandas.DataFrame
        Smart Tagging instances.
    kickoffs : tuple of int
        Detected kick-off frame of each half.
    features : pandas.DataFrame
        Team features (features/team.py), one row per frame.
    cues : pandas.DataFrame
        Dead-ball cues (features/deadball.py), one row per frame.
    """
    tracking: Tracking
    tags: pd.DataFrame
    kickoffs: tuple                 # detected kick-off frame of each half
    features: pd.DataFrame          # team features (features/team.py), one row per frame
    cues: pd.DataFrame              # dead-ball cues (features/deadball.py), one row per frame

    @property
    def timeline(self) -> Timeline:
        """The graded halves with the detected kick-offs."""
        return Timeline(halves=GRADED_HALVES, kickoffs=self.kickoffs)

    @property
    def player_speed(self) -> np.ndarray:
        """Median player speed, 1 s rolling mean (the `med_spd` feature), (n,) m/s."""
        return self.features["med_spd"].to_numpy()

    @property
    def sp_open(self) -> np.ndarray:
        """Inside a Smart Tagging SET PIECES window, before its detected restart, (n,) bool."""
        return self.cues["db_sp_open"].to_numpy() > 0.5

    def head_table(self) -> HeadTable:
        """The dead/alive head's input: shape view and tag-free cues (float32), no labels."""
        return HeadTable(shape=self.features[team.shape_columns(self.features)].reset_index(drop=True).astype(np.float32),
                         cues=deadball.tag_free_cues(self.cues).astype(np.float32))


def load_challenge_match(cfg: Config) -> ChallengeMatch:
    """Load and featurise the challenge match (cached in paths.cache).

    Reads the ATD and the Smart Tagging, detects the kick-offs, and builds the team features and the
    dead-ball cues (with the ``stage1.cues`` ball-out refinements). The feature tables are cached, keyed
    on the source of the feature modules, the three input files and ``stage1.cues``.

    Parameters
    ----------
    cfg : Config
        Uses ``paths`` and ``stage1.cues``.

    Returns
    -------
    ChallengeMatch

    Raises
    ------
    FileNotFoundError
        If a file of the organisers' bundle is missing.
    """
    paths = cfg.paths
    trk = tracking_io.load_tracking(paths.match_data, paths.cache)
    tags = tracking_io.load_smart_tagging(paths.match_data)
    with timeline.use(timeline.Timeline(halves=GRADED_HALVES)):
        kickoffs = observed.detect_kickoffs(trk, tags)
    log.info("kick-offs detected at frames %s", kickoffs)
    key = cache.fingerprint(*FEATURE_CODE, sys.modules[__name__],
                            *(tracking_io.asset_path(paths.match_data, a) for a in ("atd_raw", "atd_meta", "smart_tagging")),
                            cfg.stage1.cues)

    def build() -> dict:
        """The team features and dead-ball cues of the challenge match."""
        with timeline.use(timeline.Timeline(halves=GRADED_HALVES, kickoffs=kickoffs)):
            sign = common.attack_sign_by_half(trk.n_frames, trk.home_attacks_positive_x_first_half)
            rules = observed.OutRules(cfg.stage1.cues.flight_veto, cfg.stage1.cues.inward_throw)
            return {"features": team.team_features(trk, sign), "cues": deadball.dead_ball_cues(trk, tags, rules)}

    tables = cache.cached(paths.cache, "challenge_match_tables", key, build)
    return ChallengeMatch(tracking=trk, tags=tags, kickoffs=kickoffs, features=tables["features"], cues=tables["cues"])
