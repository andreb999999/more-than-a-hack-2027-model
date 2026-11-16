"""Forced dead-ball intervals: where the match itself says the ball is dead, D is not a guess.

The union of these intervals is forced to D by stage 1's decoder:

    pre_kickoff   graded frames before each half's detected kick-off.
    ball_out      the observed rules (rules.observed.observed_stoppages): ball out / vanished at a line / at rest /
                  a player outside, each held until the restart.
    goal          from each goal to the kick-off that follows it. A Smart Tagging GOALS interval is
                  expected to be followed by a "Kick Off (After Goal)" set piece. Camera tracking typically
                  loses the ball at the goal mouth, so the goal is anchored at the last frame where the ball is
                  seen in the box the scoring team attacks and then stays untracked for >= 2 s, with no
                  certain play between that frame and the kick-off window (fallback: the end of the last
                  certain play). The kick-off is the first strike near the centre spot after the window
                  opens, else the window start + 3 s.
    set_piece     each Smart Tagging SET PIECES window, from its start to its detected restart (the cue
                  `db_sp_open`), cut at the first certain-play frame.

Every interval ends at a detected restart (or at certain play), so possessions start there.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import timeline
from ..constants import FPS, HOME_TAG, PITCH_LENGTH, PITCH_WIDTH
from ..features.common import runs
from ..io.tracking import Tracking
from . import observed
from .kinematics import ball_speed

KICKOFF_DELAY_S = 3.0          # restart delay after a Kick Off SET PIECES window starts, when no strike is seen
GOAL_SEARCH_BEFORE_S = 60.0    # the scoring attack can start before the GOALS tag
VANISH_S = 2.0                 # ball untracked this long after its last sighting in the box
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16
CENTRE_SPOT_M = 3.0
STRIKE_MPS = 4.0


@dataclass
class Interval:
    """One forced dead-ball interval.

    Attributes
    ----------
    start, stop : int
        Half-open video-frame range.
    kind : str
        "pre_kickoff", "ball_out", "goal" or "set_piece".
    note : str
        How the interval was found (e.g. "cut at certain play"); for goals it starts with the scoring team.
    team : str
        Goals only: the scoring team, as named in the Smart Tagging.
    """
    start: int
    stop: int        # half-open
    kind: str
    note: str = ""
    team: str = ""   # goals: the scoring team, as named in the Smart Tagging


def _centre_strike(tracking: Tracking, start: int, stop: int, sustain: int = 3) -> int | None:
    """First frame in [start, stop) where the ball, within CENTRE_SPOT_M of the centre spot, moves faster
    than STRIKE_MPS for `sustain` frames (a kick-off), or None."""
    ball = tracking.ball[start:stop].astype(np.float64)
    speed = ball_speed(tracking.ball)[start:stop]
    with np.errstate(invalid="ignore"):
        near = np.hypot(ball[:, 0] - PITCH_LENGTH / 2, ball[:, 1] - PITCH_WIDTH / 2) < CENTRE_SPOT_M
        hit = near & (np.nan_to_num(speed, nan=0.0) > STRIKE_MPS)
    run = np.convolve(hit.astype(np.int8), np.ones(sustain, dtype=np.int8), "full")[sustain - 1:]
    found = np.flatnonzero(run >= sustain)
    return start + int(found[0]) if len(found) else None


def _attacks_positive_x(tracking: Tracking, team_name: str, frame: int) -> bool:
    """Whether the team named in a tag attacks towards +x at `frame` (teams swap at half-time)."""
    home_positive = tracking.home_attacks_positive_x_first_half
    if frame >= timeline.halves()[1][0]:
        home_positive = not home_positive
    return home_positive if team_name == HOME_TAG else not home_positive


def _goal_anchor(tracking: Tracking, goal, lo: int, window: int, play: np.ndarray) -> tuple[int | None, str | None]:
    """(onset frame, how) of the dead ball after a goal, or (None, None)."""
    ball = tracking.ball.astype(np.float64)
    visible = np.isfinite(ball[:, 0])
    with np.errstate(invalid="ignore"):
        if _attacks_positive_x(tracking, goal.team, goal.start_frame):
            box = ball[:, 0] > PITCH_LENGTH - BOX_DEPTH_M
        else:
            box = ball[:, 0] < BOX_DEPTH_M
        box &= np.abs(ball[:, 1] - PITCH_WIDTH / 2) < BOX_HALF_WIDTH_M
    gap = int(VANISH_S * FPS)
    later_play = np.flatnonzero(play[lo:window])
    last_play = lo + int(later_play[-1]) if len(later_play) else -1
    for frame in (np.flatnonzero(box[lo:window]) + lo)[::-1]:
        if frame <= last_play:
            break
        if not visible[frame + 1:frame + 1 + gap].any():
            return int(frame) + 1, "ball vanished in the box"
    if last_play >= 0:
        return last_play + 1, "end of last certain play"
    return None, None


def goal_stoppages(tracking: Tracking, tags: pd.DataFrame, play: np.ndarray) -> list[Interval]:
    """[Interval] from each goal to the kick-off that follows it; `note` starts with the scoring team.

    The goal is anchored at the last frame the ball is seen in the box the scoring team attacks before
    it stays untracked for VANISH_S, searched from GOAL_SEARCH_BEFORE_S before the GOALS tag to the
    following "Kick Off (After Goal)" window and after the last certain play (fallback: the end of that
    certain play). The kick-off is the first centre-spot strike within 30 s of the window start, else
    the window start + KICKOFF_DELAY_S. Goals with no following kick-off tag or no anchor are skipped.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    tags : pandas.DataFrame
        Smart Tagging table.
    play : numpy.ndarray
        (n_frames,) bool certain play (``observed.certain_play``).

    Returns
    -------
    list of Interval
        Kind "goal", with `team` set to the scoring team.
    """
    kickoffs = tags[(tags.code == "SET PIECES") & tags.type.str.startswith("Kick Off (After")]
    out = []
    for goal in tags[tags.code == "GOALS"].sort_values("start_frame").itertuples():
        following = kickoffs[kickoffs.start_frame > goal.start_frame]
        if not len(following):
            continue
        window = int(following.start_frame.min())
        strike = _centre_strike(tracking, window, window + int(30 * FPS))
        restart = strike if strike is not None else window + int(KICKOFF_DELAY_S * FPS)
        lo = max(goal.start_frame - int(GOAL_SEARCH_BEFORE_S * FPS), 0)
        onset, how = _goal_anchor(tracking, goal, lo, window, play)
        if onset is None:
            continue                     # nothing to anchor the goal on: leave it to the models
        kick = "strike seen" if strike is not None else "window + 3 s"
        out.append(Interval(onset, restart, "goal", f"{goal.team} goal ({how}), kick-off {kick}", team=goal.team))
    return out


def set_piece_stoppages(sp_open: np.ndarray, play: np.ndarray) -> list[Interval]:
    """[Interval] from each SET PIECES window start to its detected restart (`sp_open`, the cue
    db_sp_open > 0.5), never into certain play.

    Parameters
    ----------
    sp_open : numpy.ndarray
        (n_frames,) bool, inside a SET PIECES window before its detected restart.
    play : numpy.ndarray
        (n_frames,) bool certain play (``observed.certain_play``).

    Returns
    -------
    list of Interval
        Kind "set_piece", one per run of `sp_open`, cut at its first certain-play frame (note "cut at
        certain play"); runs that start in certain play are dropped.
    """
    out = []
    for a, b in runs(sp_open):
        live = np.flatnonzero(play[a:b])
        stop = a + int(live[0]) if len(live) else b
        if stop > a:
            out.append(Interval(a, stop, "set_piece", "cut at certain play" if len(live) else ""))
    return out


def intervals(tracking: Tracking, tags: pd.DataFrame, player_speed: np.ndarray, sp_open: np.ndarray) -> list[Interval]:
    """All forced dead-ball intervals of the challenge match, by kind.

    Must run inside a timeline with known kick-offs.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    tags : pandas.DataFrame
        Smart Tagging table.
    player_speed : numpy.ndarray
        (n_frames,) median player speed in m/s.
    sp_open : numpy.ndarray
        (n_frames,) bool, inside a SET PIECES window before its detected restart.

    Returns
    -------
    list of Interval
        The pre_kickoff, ball_out (``observed.observed_stoppages`` runs after each kick-off), goal and
        set_piece intervals, in that order; they may overlap.
    """
    kicks = timeline.kickoffs()
    out = [Interval(a, k, "pre_kickoff") for (a, _), k in zip(timeline.halves(), kicks) if k > a]
    mask = observed.observed_stoppages(tracking, player_speed=player_speed, tags=tags)
    for (a, _), k in zip(timeline.halves(), kicks):
        mask[a:k] = False
    out += [Interval(a, b, "ball_out") for a, b in runs(mask)]
    play = observed.certain_play(tracking)
    out += goal_stoppages(tracking, tags, play)
    out += set_piece_stoppages(sp_open, play)
    return out


def forced_dead_mask(tracking: Tracking, tags: pd.DataFrame, player_speed: np.ndarray, sp_open: np.ndarray) -> np.ndarray:
    """Boolean forced-D mask over all frames (graded frames only).

    Parameters
    ----------
    tracking, tags, player_speed, sp_open
        As in `intervals`.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, the union of all forced intervals restricted to the graded halves.
    """
    out = np.zeros(tracking.n_frames, dtype=bool)
    for iv in intervals(tracking, tags, player_speed, sp_open):
        out[iv.start:iv.stop] = True
    return out & timeline.graded_mask(tracking.n_frames)
