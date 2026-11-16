"""Observed dead-ball and live-play rules: what the tracking shows directly, with no learning.

The ATD is in pitch coordinates, so "outside the pitch" is read straight from the ball position in
metres. Every threshold is a physical scale (metres, m/s, seconds) written next to its rule.

Dead-ball evidence (R: dead ball, L: live play):
    R1  outside play: every frame before each half's detected kick-off and after the half ends.
    R2  ball out: the ball visible more than 0.25 m beyond a line (minus airborne passes the camera
        projects beyond a touchline, R2a).
    R3  ball vanished at a line: lost within 1.5 m of a line while moving towards it, missing >= 2 s,
        while the players slow down (minus a goalkeeper catching it at the goal line, R3a).
    R4  until the restart: each R2/R3/R5/R6 run is held until the restart is taken: a settled ball
        struck inside the pitch (R4), a strike from the line (R4a), a set-piece formation breaking up
        (R4b), only while the ball is seen (R4c), the ball visibly played again (R4d), a throw seen in
        mid-air (R4e), or, for a set-piece window, play visibly resumed (R4f).
    R5  ball at rest: visible and slower than 0.5 m/s for >= 2 s.
    R6  player outside: a player > 1 m beyond a line while the ball is slow or untracked, for 1 s.
    R7  everyone still: the players' median and 90th-percentile speeds below 1.0 / 1.5 m/s for 3 s
        (the strict 0.8 m/s version is a directly observed stoppage).
Live-play evidence:
    L1  certain play: the ball faster than 4 m/s inside the pitch while the fastest players sprint.
    L2  just restarted: 2 s after every observed restart strike.
    L3  keeper holds: the ball within 1.5 m of a goalkeeper in his penalty area, no ball out in 10 s.
    L4  player carries: the ball within 1 m of any player, no ball out in 10 s (soft cue only).
    L5  keeper holds an untracked ball (applied after stage 1's decode, see pipeline/stage1.py).
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import timeline
from ..constants import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W
from ..features.common import row_quantile, runs
from ..io.tracking import Tracking
from .formations import corner_formation, halves_split, kickoff_split
from .kinematics import ball_speed, distance_outside, fastest_players_speed, keeper_distance, player_speeds

OUT_MARGIN_M = 0.25

# R2a: an airborne ball projected beyond a touchline for a moment
FLIGHT_MAX_S, FLIGHT_MIN_MPS = 2.0, 1.5
# R3a: goalkeeper catch at the goal line
KEEPER_M, CARRY_MPS, REST_MPS, RELEASE_MPS, RELEASE_S = 2.0, 0.8, 0.5, 4.0, 4.0
# R7
STILL_MED_MPS, STILL_P90_MPS, STILL_MIN_S = 1.0, 1.5, 3.0
STILL_STRICT_MED_MPS = 0.8
# R1: kick-off detection
CENTRE_RADIUS_M, KICK_MPS, KICK_SUSTAIN = 4.0, 4.0, 3
KICKOFF_TAG_SEARCH_S, KICKOFF_TAG_DELAY_S, KICKOFF_SEARCH_S = 30.0, 3.0, 300.0
# R4b: a set-piece formation held 1 s, then broken for 0.5 s
FORMATION_HOLD_S, FORMATION_BROKEN_S = 1.0, 0.5
# R4d / R4f: ball visibly played inside the pitch
BALL_IN_PLAY_S = 1.5
PLAY_MPS, PLAY_INSIDE_M, PLAY_WINDOW_S = 4.0, 2.0, 4.0
RESUME_P90_MPS = 3.0
# L5
KEEPER_RELEASE_MPS, KEEPER_NEAR_M, BOX_SLACK_M = 10.0, 2.5, 2.0


@dataclass(frozen=True)
class OutRules:
    """Which refinements of the ball-out rules are on.

    The forced and observed layers always use both; the dead-ball cue features use neither (config
    `stage1.cues`).

    Attributes
    ----------
    flight_veto : bool
        R2a: an airborne ball projected beyond a line for a moment is not out (`ball_out_mask`).
    inward_throw : bool
        R4e: a ball that rested untracked and is first seen in flight into the pitch is the throw-in
        itself (`extend_to_restart`).
    """
    flight_veto: bool = True    # R2a
    inward_throw: bool = True   # R4e


ALL_RULES = OutRules()


# -- ball out of the pitch ------------------------------------------------------------------------

def _flight_through(ball: np.ndarray, mask: np.ndarray, margin: float = OUT_MARGIN_M) -> np.ndarray:
    """R2a: drop the out-runs of an airborne ball. A long ball in the air is projected beyond the
    touchline by the camera homography; a ball that really went out is stopped, caught or rolls to rest.
    A run is dropped when it lasts under FLIGHT_MAX_S, the ball is seen on every frame of it, comes back
    inside right after, and never slows below FLIGHT_MIN_MPS from 0.5 s before to 0.5 s after it."""
    speed = ball_speed(ball)
    dist = distance_outside(ball)
    out = mask.copy()
    half = FPS // 2
    for a, b in runs(mask):
        if b - a >= FLIGHT_MAX_S * FPS or b + half >= len(ball) or a < half:
            continue
        seg = speed[a - half:b + half]
        with np.errstate(invalid="ignore"):
            if np.isfinite(seg).all() and (seg >= FLIGHT_MIN_MPS).all() and (dist[b:b + half] <= margin).all():
                out[a:b] = False
    return out


def ball_out_mask(tracking: Tracking, margin: float = OUT_MARGIN_M, graded_only: bool = True,
                  rules: OutRules = ALL_RULES) -> np.ndarray:
    """R2: ball visibly outside the pitch by more than `margin` metres (minus R2a fly-throughs).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    margin : float, optional
        Metres beyond the nearest line.
    graded_only : bool, optional
        Restrict the mask to the graded frames of the active match.
    rules : OutRules, optional
        With ``rules.flight_veto``, drop the out-runs of an airborne ball (R2a).

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool.
    """
    with np.errstate(invalid="ignore"):
        mask = distance_outside(tracking.ball) > margin
    if rules.flight_veto:
        mask = _flight_through(tracking.ball.astype(np.float64), mask, margin)
    if graded_only:
        mask &= timeline.graded_mask(len(mask))
    return mask


def keeper_catch(tracking: Tracking, start: int, stop: int) -> bool:
    """R3a: the ball vanished at a goal line because the goalkeeper caught it.

    All of: the keeper within KEEPER_M of the ball where it was last seen; when it is seen again (on at
    least half of its first second), within KEEPER_M of a keeper (median over that second) inside a
    penalty area and moving (median speed over its first second > CARRY_MPS: carried, not placed); then
    released (faster than RELEASE_MPS within RELEASE_S) without first lying at rest for 1 s (a goal kick
    is placed and rests before it is struck).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    start, stop : int
        The ball gap, half-open: last seen at ``start - 1``, seen again at `stop`.

    Returns
    -------
    bool
        True if the gap is explained by a goalkeeper catch.
    """
    n = len(tracking.ball)
    if start < 1 or stop >= n or keeper_distance(tracking, start - 1) > KEEPER_M:
        return False
    ball = tracking.ball.astype(np.float64)
    first = slice(stop, min(stop + FPS, n))
    seen = np.isfinite(ball[first, 0])
    if seen.sum() < FPS // 2:
        return False
    near = np.median([keeper_distance(tracking, f) for f in range(first.start, first.stop) if np.isfinite(ball[f, 0])])
    b = ball[stop]
    inside_box = (min(b[0], L - b[0]) < 16.5) and abs(b[1] - W / 2) < 20.2
    speed = ball_speed(tracking.ball)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        moving = np.nanmedian(speed[first]) > CARRY_MPS
    if not (near < KEEPER_M and inside_box and moving):
        return False
    after = speed[stop:min(stop + int(RELEASE_S * FPS), n)]
    with np.errstate(invalid="ignore"):
        released = np.flatnonzero(after > RELEASE_MPS)
        rest = after < REST_MPS
    if not len(released):
        return False
    before = rest[:released[0]].astype(np.int8)
    rested = (np.convolve(before, np.ones(FPS, dtype=np.int8), "valid") >= FPS).any() if len(before) >= FPS else False
    return not rested


def ball_vanished_at_line(tracking: Tracking, player_speed: np.ndarray | None = None, line_m: float = 1.5,
                          min_gap_s: float = 2.0, approach_frames: int = 10, slow_mps: float = 1.6) -> np.ndarray:
    """R3: the ball last seen within `line_m` of a line (or beyond it), heading out, then missing for at
    least `min_gap_s`.

    With `player_speed` (per-frame median player speed) the players must also slow below `slow_mps` over
    the gap (median over the gap), which separates a ball that went out from one lost on the wing in play.
    Gaps at a goal line explained by a keeper catch (R3a) are dropped, and so are gaps that start outside
    graded play (pre-match, half-time).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    player_speed : numpy.ndarray, optional
        (n_frames,) median player speed in m/s; None skips the slow-down condition.
    line_m : float, optional
        Metres from a line within which the ball was last seen.
    min_gap_s : float, optional
        Minimum gap length in seconds.
    approach_frames : int, optional
        "Heading out" compares the last position with the one this many frames before it.
    slow_mps : float, optional
        Players' median speed over the gap must not exceed this (m/s).

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool mask over the qualifying gaps, graded frames only.
    """
    ball = tracking.ball.astype(np.float64)
    visible = np.isfinite(ball[:, 0])
    min_gap = int(min_gap_s * FPS)
    graded = timeline.graded_mask(len(ball))
    out = np.zeros(len(ball), dtype=bool)
    for start, stop in runs(~visible):
        if start < approach_frames or stop - start < min_gap or not graded[start]:
            continue   # a gap that starts before graded play (pre-match, half-time) is no evidence
        last = ball[start - 1]
        before = ball[start - approach_frames]
        if not (np.isfinite(last).all() and np.isfinite(before).all()):
            continue
        # distance to each line from the inside, and whether the ball was moving towards it
        to_line = np.array([last[0], L - last[0], last[1], W - last[1]])
        towards = np.array([last[0] < before[0], last[0] > before[0], last[1] < before[1], last[1] > before[1]])
        if not ((to_line < line_m) & towards).any():
            continue
        if player_speed is not None and np.nanmedian(player_speed[start:stop]) > slow_mps:
            continue
        at_goal_line = ((to_line[:2] < line_m) & towards[:2]).any()
        if not (at_goal_line and keeper_catch(tracking, start, stop)):
            out[start:stop] = True
    return out & timeline.graded_mask(len(out))


def ball_lost_while_slow(tracking: Tracking, player_speed: np.ndarray, min_gap_s: float = 3.0,
                         slow_mps: float = 1.2) -> np.ndarray:
    """Soft dead-ball cue (never forced): the ball untracked for at least `min_gap_s` anywhere on the pitch
    while the players' median speed over the gap is below `slow_mps`.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    player_speed : numpy.ndarray
        (n_frames,) median player speed in m/s.
    min_gap_s : float, optional
        Minimum gap length in seconds.
    slow_mps : float, optional
        Speed threshold in m/s.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool mask over the qualifying gaps, graded frames only.
    """
    visible = np.isfinite(tracking.ball[:, 0])
    out = np.zeros(len(visible), dtype=bool)
    speed = np.asarray(player_speed, dtype=np.float64)
    for start, stop in runs(~visible):
        if stop - start < min_gap_s * FPS:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if np.nanmedian(speed[start:stop]) < slow_mps:
                out[start:stop] = True
    return out & timeline.graded_mask(len(out))


def ball_with_player(tracking: Tracking, keeper_only: bool, reach_m: float = 1.5, clean_s: float = 10.0) -> np.ndarray:
    """L3 / L4 (soft alive cues): the ball visible within `reach_m` of a player (keeper_only: a
    goalkeeper, with the ball inside a penalty area) and not out of the pitch in the previous `clean_s`.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    keeper_only : bool
        L3 (goalkeepers only, ball inside either penalty area) if True, else L4 (any player).
    reach_m : float, optional
        Ball-to-player distance in metres.
    clean_s : float, optional
        Seconds (trailing window, current frame included) without the ball more than OUT_MARGIN_M
        beyond a line.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, all frames (not restricted to the graded halves).
    """
    ball = tracking.ball.astype(np.float64)
    pos = tracking.players.astype(np.float64)
    d = np.hypot(pos[..., 0] - ball[:, None, 0], pos[..., 1] - ball[:, None, 1])
    with np.errstate(invalid="ignore"):
        if keeper_only:
            gk = np.asarray(tracking.goalkeeper, dtype=bool)
            d = np.where(gk[None, :], d, np.nan)
            area = (np.minimum(ball[:, 0], L - ball[:, 0]) < 16.5) & (np.abs(ball[:, 1] - W / 2) < 20.2)
        else:
            area = np.ones(len(ball), dtype=bool)
        near = (np.where(np.isfinite(d), d, np.inf).min(1) < reach_m) & area
        out = distance_outside(tracking.ball) > OUT_MARGIN_M
    recent = pd.Series(out.astype(np.float64)).rolling(int(clean_s * FPS), min_periods=1).max().to_numpy() > 0
    return near & ~recent


def certain_play(tracking: Tracking, ball_mps: float = 4.0, sprint_mps: float = 4.5, hold_s: float = 1.0) -> np.ndarray:
    """L1: frames that are unmistakably live: the ball faster than `ball_mps` inside the pitch for
    `hold_s` while the fastest players (90th percentile speed, `hold_s` median) sprint faster than
    `sprint_mps`; the conjunction itself must hold for `hold_s`.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    ball_mps : float, optional
        Ball speed threshold in m/s.
    sprint_mps : float, optional
        Threshold of the players' 90th-percentile speed in m/s (per-frame steps above 12 m/s are slot
        switches and ignored).
    hold_s : float, optional
        Length in seconds of the centred windows.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, all frames (not restricted to the graded halves).
    """
    window = int(hold_s * FPS)
    pos = tracking.players.astype(np.float64)
    step = np.sqrt((np.diff(pos, axis=0) ** 2).sum(-1)) * FPS
    step = np.vstack([np.full((1, step.shape[1]), np.nan), step])
    step[step > 12.0] = np.nan                         # slot switches teleport; nobody runs 12 m/s
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p90 = np.nanpercentile(step, 90, axis=1)
        sprint = pd.Series(p90).rolling(window, center=True, min_periods=1).median().to_numpy() > sprint_mps
        moving = (np.nan_to_num(ball_speed(tracking.ball), nan=0.0) > ball_mps) & (distance_outside(tracking.ball) < 0)
    held = pd.Series(moving.astype(np.float64)).rolling(window, center=True, min_periods=1).min().to_numpy() > 0.5
    both = (held & sprint).astype(np.float64)
    return pd.Series(both).rolling(window, center=True, min_periods=1).min().to_numpy() > 0.5


def ball_at_rest(tracking: Tracking, min_s: float = 2.0, rest_mps: float = 0.5) -> np.ndarray:
    """R5: the ball visible and slower than `rest_mps` for at least `min_s`, wherever it lies.

    The whole run is marked: in open play somebody touches a ball within a second or two.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    min_s : float, optional
        Minimum run length in seconds.
    rest_mps : float, optional
        Ball speed threshold in m/s.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, graded frames only.
    """
    speed = ball_speed(tracking.ball)
    with np.errstate(invalid="ignore"):
        still = speed < rest_mps
    out = np.zeros(len(still), dtype=bool)
    for start, stop in runs(still):
        if stop - start >= min_s * FPS:
            out[start:stop] = True
    return out & timeline.graded_mask(len(out))


def player_outside(tracking: Tracking, margin: float = 1.0, hold_s: float = 1.0, ball_mps: float = 2.0) -> np.ndarray:
    """R6: a player stands more than `margin` metres beyond a line while the ball is slow (< `ball_mps`)
    or untracked, held for `hold_s` (a throw-in taker, a player retrieving the ball).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    margin : float, optional
        Metres beyond a line.
    hold_s : float, optional
        Seconds the condition must hold (centred window).
    ball_mps : float, optional
        Ball speed threshold in m/s.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, graded frames only.
    """
    pos = tracking.players.astype(np.float64)
    with np.errstate(invalid="ignore"):
        beyond = np.maximum.reduce([-pos[..., 0], pos[..., 0] - L, -pos[..., 1], pos[..., 1] - W])
        anyone = (beyond > margin).any(1)
        slow = ~(ball_speed(tracking.ball) >= ball_mps)          # NaN (untracked) counts as slow
    both = (anyone & slow).astype(np.float64)
    held = pd.Series(both).rolling(int(hold_s * FPS), center=True, min_periods=1).min().to_numpy() > 0.5
    return held & timeline.graded_mask(len(held))


def everyone_still(tracking: Tracking, med_mps: float = STILL_MED_MPS, p90_mps: float = STILL_P90_MPS,
                   min_s: float = STILL_MIN_S) -> np.ndarray:
    """R7: the players' median speed below `med_mps` and their 90th percentile below `p90_mps` (1 s
    rolling medians) for at least `min_s`; the whole run is D.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    med_mps : float, optional
        Threshold of the median player speed in m/s (STILL_STRICT_MED_MPS in the strict version).
    p90_mps : float, optional
        Threshold of the 90th-percentile player speed in m/s.
    min_s : float, optional
        Minimum run length in seconds.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, graded frames only.
    """
    ps = player_speeds(tracking)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p90 = pd.Series(row_quantile(ps, 0.9)).rolling(FPS, center=True, min_periods=1).median().to_numpy()
        med = pd.Series(row_quantile(ps, 0.5)).rolling(FPS, center=True, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore"):
        still = (p90 < p90_mps) & (med < med_mps)
    out = np.zeros(len(still), dtype=bool)
    for start, stop in runs(still):
        if stop - start >= min_s * FPS:
            out[start:stop] = True
    return out & timeline.graded_mask(len(out))


def certain_dead(tracking: Tracking) -> np.ndarray:
    """Directly observed D: R2 (ball visibly out), R5 (ball at rest), R6 (player outside, ball slow) and
    strict R7 (everyone very slow). Forced to D in every decode, never trimmed by the playbook.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, graded frames only.
    """
    return (ball_out_mask(tracking) | ball_at_rest(tracking) | player_outside(tracking)
            | everyone_still(tracking, med_mps=STILL_STRICT_MED_MPS))


# -- kick-offs (R1) ----------------------------------------------------------------------------------

def _centre_kick(ball: np.ndarray, speed: np.ndarray, lo: int, hi: int, tracking: Tracking | None = None) -> int | None:
    """First frame in [lo, hi) of a ball near the centre spot moving faster than KICK_MPS for
    KICK_SUSTAIN frames. With `tracking`, the players must also stand in the kick-off formation 1 s
    before (used when no tag says a kick-off is due)."""
    hi = min(hi, len(ball))
    if hi <= lo:
        return None
    b = ball[lo:hi].astype(np.float64)
    with np.errstate(invalid="ignore"):
        near = np.hypot(b[:, 0] - L / 2, b[:, 1] - W / 2) < CENTRE_RADIUS_M
        fast = np.nan_to_num(speed[lo:hi], nan=0.0) > KICK_MPS
    run = np.convolve((near & fast).astype(np.int8), np.ones(KICK_SUSTAIN, dtype=np.int8))[KICK_SUSTAIN - 1:]
    for i in np.flatnonzero(run >= KICK_SUSTAIN):
        f = lo + int(i)
        if tracking is None or halves_split(tracking.players[max(f - FPS, 0)], tracking.team, tracking.goalkeeper):
            return f
    return None


def detect_kickoffs(tracking: Tracking, tags: pd.DataFrame | None = None) -> tuple[int, ...]:
    """R1: the kick-off frame of each graded half of the active match, detected from the data.

    With Smart Tagging: the 'Kick Off (Start)' SET PIECES tag of the half opens a 30 s search for the
    strike near the centre spot (ball < 4 m from the spot, > 4 m/s for 3 frames); no strike seen -> tag
    start + 3 s. Without tags (DFL), or without a kick-off tag for the half: the first centre-spot strike
    in the half's first 5 minutes with both teams in their own halves 1 s before, else the half's first
    frame. A kick-off is never placed before its half starts.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    tags : pandas.DataFrame, optional
        Smart Tagging table; None or empty for DFL.

    Returns
    -------
    tuple of int
        One kick-off frame per graded half of the active timeline.
    """
    out = []
    starts = None
    speed = ball_speed(tracking.ball)
    if tags is not None and len(tags):
        kick = tags[(tags.code == "SET PIECES") & tags.type.str.lower().str.startswith("kick off (start")]
        starts = np.sort(kick.start_frame.to_numpy())
    for a, b in timeline.halves():
        frame = None
        if starts is not None:
            near = starts[(starts >= a - int(KICKOFF_TAG_SEARCH_S * FPS)) & (starts < b)]
            if len(near):
                window = int(near[0])
                frame = _centre_kick(tracking.ball, speed, max(window, a), window + int(KICKOFF_TAG_SEARCH_S * FPS))
                if frame is None:
                    frame = window + int(KICKOFF_TAG_DELAY_S * FPS)
        if frame is None:
            frame = _centre_kick(tracking.ball, speed, a, a + int(KICKOFF_SEARCH_S * FPS), tracking)
        out.append(max(a, frame if frame is not None else a))
    return tuple(out)


def pre_kickoff_mask(tracking: Tracking, tags: pd.DataFrame | None = None) -> np.ndarray:
    """R1: every frame outside play is D: from the start of the video, and from the end of each graded
    half, until the next detected kick-off.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    tags : pandas.DataFrame, optional
        Smart Tagging table, passed to `detect_kickoffs`.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, True everywhere except from each kick-off to the end of its half (frames after
        the last half included).
    """
    out = np.ones(tracking.n_frames, dtype=bool)
    for (_, stop), kickoff in zip(timeline.halves(), detect_kickoffs(tracking, tags)):
        out[kickoff:stop] = False
    return out


# -- restarts (R4) ---------------------------------------------------------------------------------

def ball_played(ball: np.ndarray) -> np.ndarray:
    """The ball visible more than PLAY_INSIDE_M inside every line and faster than PLAY_MPS.

    Parameters
    ----------
    ball : numpy.ndarray
        (n_frames, 2) ball position in metres, NaN where missing.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool.
    """
    with np.errstate(invalid="ignore"):
        return (np.nan_to_num(ball_speed(ball), nan=0.0) > PLAY_MPS) & (distance_outside(ball) < -PLAY_INSIDE_M)


def play_resumed_onsets(tracking: Tracking) -> np.ndarray:
    """R4f: frames where play visibly resumed: the first frame of every PLAY_WINDOW_S window holding a
    cumulative BALL_IN_PLAY_S of frames with the ball played while the fastest players move (90th
    percentile speed > RESUME_P90_MPS). The onset, so a stoppage ends where the ball was first played.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, True on the first played frame of each trailing window that reaches the
        cumulative threshold.
    """
    need = int(BALL_IN_PLAY_S * FPS)
    p90 = fastest_players_speed(tracking)
    with np.errstate(invalid="ignore"):
        played = ball_played(tracking.ball) & (p90 > RESUME_P90_MPS)
    window = int(PLAY_WINDOW_S * FPS)
    share = pd.Series(played.astype(np.float64)).rolling(window, min_periods=1).sum().to_numpy()
    out = np.zeros(len(played), dtype=bool)
    full = np.flatnonzero((share >= need) & ~np.r_[False, share[:-1] >= need])
    for t in full:
        lo = max(t - window + 1, 0)
        first = np.flatnonzero(played[lo:t + 1])
        out[lo + int(first[0]) if len(first) else t] = True
    return out


def setpiece_formation(tracking: Tracking) -> np.ndarray:
    """R4b: a set-piece formation whose break-up means the restart was taken even when the ball is not
    tracked: the kick-off split (Law 8) or a corner (player at the flag and a crowded penalty area).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool: `formations.kickoff_split` or ``corner_formation > 0.5``.
    """
    return kickoff_split(tracking) | (corner_formation(tracking) > 0.5)


def extend_to_restart(mask: np.ndarray, ball: np.ndarray, player_speed: np.ndarray | None = None,
                      play: np.ndarray | None = None, formation: np.ndarray | None = None,
                      restart_mps: float = 6.0, sustain: int = 3, settle_mps: float = 2.0, settle_s: float = 1.0,
                      max_s: float = 60.0, live_mps: float = 3.0, live_s: float = 2.0, spot_m: float = 3.0,
                      rules: OutRules = ALL_RULES) -> tuple[np.ndarray, np.ndarray]:
    """R4: keep D after each run of `mask` until the restart is taken.

    A restart is a ball that first settles (slower than `settle_mps`, or untracked, for `settle_s`:
    placed for a goal kick or corner, held for a throw-in) and is then struck inside the pitch (faster
    than `restart_mps` for `sustain` frames); requiring the settle phase stops a ball passed back to the
    taker from counting. The extension also stops when
      - the median player speed stays above `live_mps` for `live_s` (players running: play resumed),
      - certain play is reached (`play`, L1),
      - the ball is visibly played inside the pitch for a cumulative 1.5 s within 4 s (R4d),
      - a set-piece formation held for 1 s breaks up for 0.5 s (R4b, `formation`),
      - it reaches `max_s`.
    R4a: when the triggering run itself ends with the strike from a spot at most `spot_m` beyond the line
    (a throw-in or corner taken from outside), there is nothing to extend. R4e: a ball that rested
    untracked (in the thrower's hands) and is first seen already in flight into the pitch is that throw.

    Parameters
    ----------
    mask : numpy.ndarray
        (n,) bool, the triggering runs (e.g. R2 | R3 | R5 | R6).
    ball : numpy.ndarray
        (n, 2) ball position in metres, NaN where missing.
    player_speed : numpy.ndarray, optional
        (n,) median player speed in m/s; None disables the `live_mps` stop.
    play : numpy.ndarray, optional
        (n,) bool certain play (L1); None disables that stop.
    formation : numpy.ndarray, optional
        (n,) bool set-piece formation (R4b, `setpiece_formation`); None disables that stop.
    restart_mps : float, optional
        Ball speed of a restart strike (m/s).
    sustain : int, optional
        Frames the strike must last.
    settle_mps : float, optional
        A ball slower than this (or untracked) is settled (m/s).
    settle_s : float, optional
        Seconds the ball must be settled before a strike counts.
    max_s : float, optional
        Longest extension in seconds.
    live_mps : float, optional
        Median player speed (m/s) that means play resumed ...
    live_s : float, optional
        ... when held for this many seconds (centred window).
    spot_m : float, optional
        R4a / R4e: farthest restart spot beyond the line, metres.
    rules : OutRules, optional
        With ``rules.inward_throw``, apply R4e.

    Returns
    -------
    extended : numpy.ndarray
        (n,) bool, `mask` plus the extensions, graded frames only.
    restarts : numpy.ndarray
        (n_restarts, 2) int64 rows ``(frame, kind)``: kind 0 strike, 1 R4a, 2 R4b formation break.
    """
    out = mask.copy()
    speed = ball_speed(ball)
    dist = distance_outside(ball)
    with np.errstate(invalid="ignore"):
        inside = dist < 0
        struck = (np.nan_to_num(speed, nan=0.0) > restart_mps) & inside
        settled = ~(speed >= settle_mps)              # NaN (untracked) counts as settled
    if player_speed is not None:
        window = int(live_s * FPS)
        sustained = pd.Series(np.asarray(player_speed, dtype=np.float64)).rolling(
            window, center=True, min_periods=1).min().to_numpy()
        with np.errstate(invalid="ignore"):
            running = sustained > live_mps
    else:
        running = np.zeros(len(ball), dtype=bool)
    if play is not None:
        running = running | play
    with np.errstate(invalid="ignore"):                # R4d
        played = (np.nan_to_num(speed, nan=0.0) > 4.0) & (dist < -2.0)
    share = pd.Series(played.astype(np.float64)).rolling(4 * FPS, min_periods=1).sum().to_numpy()
    running = running | (share >= BALL_IN_PLAY_S * FPS)
    settle = int(settle_s * FPS)
    cap = int(max_s * FPS)
    need_form, need_break = int(FORMATION_HOLD_S * FPS), int(FORMATION_BROKEN_S * FPS)
    fast = np.nan_to_num(speed, nan=0.0) > restart_mps
    restarts = []
    for start, stop in runs(mask):
        burst = 0                                  # trailing moving frames of the run
        while stop - burst - 1 >= start and not settled[stop - burst - 1]:
            burst += 1
        rest = 0                                   # settled frames just before them (may precede the run)
        while stop - burst - rest - 1 >= max(start - cap, 0) and settled[stop - burst - rest - 1]:
            rest += 1
        spot = dist[stop - burst - rest:stop - burst]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            where = np.nanmedian(spot) if rest and np.isfinite(spot).any() else np.nan
            if not np.isfinite(where):             # untracked at rest: where it is first seen in the run
                seen = np.flatnonzero(np.isfinite(dist[start:stop]))
                where = dist[start + seen[0]] if len(seen) else np.nan
        if rules.inward_throw and burst and rest >= settle and where > spot_m:
            # R4e: rested untracked, first seen in flight a few metres out, faster than restart_mps within
            # 0.2 s of its first sighting and inside the pitch within 1 s: the throw itself
            f0 = stop - burst                      # start of the visible stretch that holds the burst
            while f0 - 1 >= start and np.isfinite(dist[f0 - 1]):
                f0 -= 1
            if f0 - settle >= 0 and not np.isfinite(dist[f0 - settle:f0]).any():
                with np.errstate(invalid="ignore"):
                    if np.nanmax(speed[f0:f0 + 5], initial=0.0) > restart_mps and (dist[f0:f0 + FPS] < 0).any():
                        where = 0.0
        # R4a: a restart is taken from the line (thrower's hands, corner arc); a ball resting further out
        # is being retrieved, and kicking it back to the taker is not the restart
        if burst and rest >= settle and where <= spot_m:
            # the smoothed ball ramps over ~1 s: the strike may reach restart_mps after the run ends
            window = fast[stop - burst:stop + FPS].astype(np.int8)
            if (np.convolve(window, np.ones(sustain, dtype=np.int8))[sustain - 1:] >= sustain).any():
                restarts.append((stop - burst, 1))
                continue
        end = min(stop + cap, len(out))
        # trailing settled frames of the triggering run count: an untracked ball at a line is settled
        moving = np.flatnonzero(~settled[start:stop])
        quiet, streak = (stop - start) - (moving[-1] + 1 if len(moving) else 0), 0
        armed = quiet >= settle
        formed = int(formation[start:stop].sum()) if formation is not None else 0
        broken, restarted = 0, None
        for t in range(stop, end):
            quiet = quiet + 1 if settled[t] else 0
            armed = armed or quiet >= settle
            streak = streak + 1 if struck[t] else 0
            if armed and streak >= sustain:
                end = restarted = t - sustain + 1
                kind = 0
                break
            if formation is not None:
                if formation[t]:
                    formed, broken = formed + 1, 0
                else:
                    broken += 1
                if formed >= need_form and broken >= need_break:
                    end = restarted = t - broken + 1
                    kind = 2
                    break
            if running[t]:
                end = t
                break
        out[stop:end] = True
        if restarted is not None:
            restarts.append((restarted, kind))
    out &= timeline.graded_mask(len(out))
    return out, np.array(restarts, dtype=np.int64).reshape(-1, 2)


def _out_evidence(tracking: Tracking, player_speed: np.ndarray | None, rules: OutRules = ALL_RULES) -> np.ndarray:
    """R2 | R3 | R5 | R6: the runs that start a stoppage held until the restart."""
    mask = ball_out_mask(tracking, rules=rules) | ball_vanished_at_line(tracking, player_speed=player_speed)
    return mask | ball_at_rest(tracking) | player_outside(tracking)


def observed_stoppages(tracking: Tracking, player_speed: np.ndarray | None = None, tags: pd.DataFrame | None = None,
               rules: OutRules = ALL_RULES) -> np.ndarray:
    """D where the observed rules are sure: R2 | R3 | R5 | R6 held until the restart (R4, kept only where
    the ball is seen within 1 s, R4c), plus R7 (everyone very slow) and R1 (outside play).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    player_speed : numpy.ndarray, optional
        (n_frames,) median player speed in m/s, used by R3 and R4.
    tags : pandas.DataFrame, optional
        Smart Tagging table, used for the kick-offs of R1.
    rules : OutRules, optional
        Ball-out refinements (R2a, R4e).

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool; includes every frame outside play (R1), graded or not.
    """
    mask = _out_evidence(tracking, player_speed, rules)
    extended, _ = extend_to_restart(mask, tracking.ball, player_speed=player_speed, play=certain_play(tracking),
                                    formation=setpiece_formation(tracking), rules=rules)
    # R4c: the 'until the restart' frames are kept only where the ball is seen within 1 s: with the ball
    # untracked the restart cannot be seen
    seen = pd.Series(np.isfinite(tracking.ball[:, 0]).astype(np.float64)).rolling(
        2 * FPS + 1, center=True, min_periods=1).max().to_numpy() > 0
    dead = extended & (mask | seen)
    dead = dead | everyone_still(tracking)
    return dead | pre_kickoff_mask(tracking, tags)


def keeper_holds_untracked(tracking: Tracking, max_gap_s: float = 25.0) -> np.ndarray:
    """L5: the goalkeeper has the ball in his hands and the tracker loses it (typically while the
    players walk; their speed is not tested).

    The ball is last seen AND seen again inside the same penalty area (2 m slack), a goalkeeper is within
    KEEPER_NEAR_M of it at one of the two moments, it is seen again inside the pitch (at least 1 m in), no
    ball out was seen in the 2 s before the gap, the gap lasts at most `max_gap_s`, and the keeper then
    releases it hard (faster than KEEPER_RELEASE_MPS within 2 s: a throw or a punt) without it first
    resting 1 s (a goal kick or free kick is placed and rests) and somebody runs after it (the players'
    90th-percentile speed above RESUME_P90_MPS within 3 s of the release).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    max_gap_s : float, optional
        Longest gap in seconds.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool mask over the qualifying gaps, graded frames only.
    """
    ball = tracking.ball.astype(np.float64)
    visible = np.isfinite(ball[:, 0])
    speed = ball_speed(tracking.ball)
    out_seen = ball_out_mask(tracking)
    p90 = fastest_players_speed(tracking)
    n = len(ball)
    res = np.zeros(n, dtype=bool)

    def box_of(xy: np.ndarray) -> int:
        """Penalty area (with slack) holding `xy`: 0 at x = 0, 1 at x = L, -1 for none."""
        x, y = xy
        if not (np.isfinite(x) and np.isfinite(y)) or abs(y - W / 2) > 20.16 + BOX_SLACK_M:
            return -1
        if x < 16.5 + BOX_SLACK_M:
            return 0
        if x > L - 16.5 - BOX_SLACK_M:
            return 1
        return -1

    for start, stop in runs(~visible):
        if start < 2 * FPS or stop >= n or stop - start > max_gap_s * FPS:
            continue
        box = box_of(ball[start - 1])
        if box < 0 or box_of(ball[stop]) != box or not distance_outside(ball[stop:stop + 1])[0] < -1.0:
            continue
        if out_seen[start - 2 * FPS:start].any():
            continue
        if min(keeper_distance(tracking, start - 1), keeper_distance(tracking, stop)) > KEEPER_NEAR_M:
            continue
        after = speed[stop:min(stop + 2 * FPS, n)]
        with np.errstate(invalid="ignore"):
            released = np.flatnonzero(after > KEEPER_RELEASE_MPS)
            rest = after < REST_MPS
        if not len(released):
            continue
        before = rest[:released[0]].astype(np.int8)
        if len(before) >= FPS and (np.convolve(before, np.ones(FPS, dtype=np.int8), "valid") >= FPS).any():
            continue
        r = stop + int(released[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if not np.nanmax(p90[r:min(r + 3 * FPS, n)], initial=0.0) > RESUME_P90_MPS:
                continue                # nobody runs after it: the keeper handed the ball to a set-piece taker
        res[start:stop] = True
    return res & timeline.graded_mask(n)


def restart_alive(tracking: Tracking, player_speed: np.ndarray | None = None, after_s: float = 2.0) -> np.ndarray:
    """Certainly live frames: L1 certain play, L3 keeper holds the ball, and `after_s` after every
    observed restart strike (L2, kind 0 only: R4a and formation breaks end the forced D but are not
    certain enough to force alive), minus the directly observed D.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    player_speed : numpy.ndarray, optional
        (n_frames,) median player speed in m/s, used to find the restarts (R3, R4).
    after_s : float, optional
        Seconds marked alive after each restart strike.

    Returns
    -------
    numpy.ndarray
        (n_frames,) bool, graded frames only.
    """
    _, restarts = extend_to_restart(_out_evidence(tracking, player_speed), tracking.ball, player_speed=player_speed,
                                    play=certain_play(tracking), formation=setpiece_formation(tracking))
    alive = certain_play(tracking) | ball_with_player(tracking, True)
    for r, kind in restarts:
        if kind == 0:
            alive[r:r + int(after_s * FPS)] = True
    return alive & ~certain_dead(tracking) & timeline.graded_mask(len(alive))
