"""Set-piece formations prescribed by the Laws of the Game, read from tracking (soft scores in [0, 1]).

Kick-off (Law 8: every player in his own half), corner (Law 17: ball in the corner arc, crowded penalty
area), free kick (Law 13: a wall of team-mates standing shoulder to shoulder ~9.15 m from the ball),
goal kick (Law 16: keeper in his goal area, opponents outside the penalty area), throw-in (Law 15:
thrower on or behind the touchline). These configurations are rare in open play, so they are evidence
of a stoppage. References: IFAB Laws of the Game 2023/24; Vidal-Codina et al. (2022), "Automatic event
detection in football using tracking data", Sports Engineering 25.
"""

import numpy as np

from ..constants import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W
from ..features.common import rolling, sigmoid
from ..io.tracking import Tracking

WALL_RING = (7.0, 11.5)   # 9.15 m +/- tracking error and players creeping in
WALL_GAP_M = 1.4          # shoulder-to-shoulder spacing plus tracking noise
DUPLICATE_M = 0.35        # same-team slots closer than this are one person tracked twice
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16      # penalty area (Law 1)
SIX_DEPTH, SIX_HALF_WIDTH = 5.5, 9.16        # goal area (Law 1)
CORNER_TAKER_M = 3.0      # taker stands at the 1 m corner arc; run-up a couple of metres
THROW_BEHIND_M = 0.2      # thrower's feet on or behind the touchline, beyond tracking noise
KICKOFF_SLACK_M = 0.5     # players standing on the halfway line


def _own_half_fraction(tracking: Tracking, sign: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per team, the fraction of tracked players standing in their own half, and the smaller head count."""
    px = tracking.players[:, :, 0].astype(np.float64)
    fractions = []
    for team_code, team_sign in ((0, 1.0), (1, -1.0)):
        cols = tracking.team == team_code
        attacking_positive = (sign * team_sign)[:, None] > 0
        x_att = np.where(attacking_positive, px[:, cols], L - px[:, cols])   # own goal at x_att = 0
        finite = np.isfinite(x_att)
        own = finite & (x_att <= L / 2 + KICKOFF_SLACK_M)
        fractions.append(own.sum(1) / np.maximum(finite.sum(1), 1))
        fractions.append(finite.sum(1))
    return fractions[0], fractions[2], np.minimum(fractions[1], fractions[3])


def kickoff_formation(tracking: Tracking, sign: np.ndarray) -> np.ndarray:
    """Both teams entirely in their own halves (Law 8): logistic in the smaller of the two own-half
    fractions around 0.93 (one stray slot allowed), needing >= 5 tracked players per team.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres (all slots, goalkeepers included).
    sign : numpy.ndarray
        (n,) +1 where the home team attacks +x, -1 otherwise.

    Returns
    -------
    numpy.ndarray
        (n,) float32 score in [0, 1], 1 s centred mean.
    """
    f_home, f_away, n_min = _own_half_fraction(tracking, sign)
    score = sigmoid((np.minimum(f_home, f_away) - 0.93) / 0.02) * sigmoid((n_min - 4.5) / 0.5)
    return rolling(score.astype(np.float32), FPS)


def kickoff_split(tracking: Tracking, slack_m: float = 2.0, share: float = 0.85, min_players: int = 5) -> np.ndarray:
    """Per frame: each team's tracked outfield players (>= `share` of them) in one half, the two teams in
    opposite halves (Law 8 kick-off formation; the attack direction is not needed).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    slack_m : float, optional
        A player within this many metres beyond the halfway line still counts for either half.
    share : float, optional
        Minimum share of a team's tracked outfield players in its half.
    min_players : int, optional
        Minimum tracked outfield players per team.

    Returns
    -------
    numpy.ndarray
        (n,) bool.
    """
    x = tracking.players[:, :, 0].astype(np.float64)
    outfield = ~np.asarray(tracking.goalkeeper, dtype=bool)
    finite = np.isfinite(x) & outfield
    sides = []
    for t in (0, 1):
        cols = (tracking.team == t)
        n = (finite & cols).sum(1)
        with np.errstate(invalid="ignore"):
            left = ((x < L / 2 + slack_m) & finite & cols).sum(1) / np.maximum(n, 1)
            right = ((x > L / 2 - slack_m) & finite & cols).sum(1) / np.maximum(n, 1)
        sides.append((left >= share, right >= share, n >= min_players))
    (hl, hr, hn), (al, ar, an) = sides
    return hn & an & ((hl & ar) | (hr & al))


def halves_split(players: np.ndarray, team: np.ndarray, keeper: np.ndarray | None = None,
                 slack_m: float = 2.0, share: float = 0.85) -> bool:
    """kickoff_split for one frame: True when at least `share` of each team's tracked outfield players
    stand in one half, the two teams in opposite halves.

    Parameters
    ----------
    players : numpy.ndarray
        (n_slots, 2) positions in metres at one frame.
    team : numpy.ndarray
        (n_slots,) 0 = home, 1 = away.
    keeper : numpy.ndarray, optional
        (n_slots,) bool goalkeepers, excluded; None counts every slot.
    slack_m : float, optional
        A player within this many metres beyond the halfway line still counts for either half.
    share : float, optional
        Minimum share of a team's tracked players in its half.

    Returns
    -------
    bool
        False also when a team has fewer than 5 tracked players.
    """
    x = players[:, 0].astype(np.float64)
    outfield = np.ones(len(x), dtype=bool) if keeper is None else ~np.asarray(keeper, dtype=bool)
    sides = []
    for t in (0, 1):
        xs = x[(team == t) & outfield & np.isfinite(x)]
        if len(xs) < 5:
            return False
        if (xs < L / 2 + slack_m).mean() >= share:
            sides.append(-1)
        elif (xs > L / 2 - slack_m).mean() >= share:
            sides.append(1)
        else:
            return False
    return sides[0] != sides[1]


def in_box(px: np.ndarray, py: np.ndarray, end: int, depth: float, half_width: float) -> np.ndarray:
    """Inside the penalty / goal area at end 0 (x = 0) or 1 (x = L).

    Parameters
    ----------
    px, py : numpy.ndarray
        Positions in metres, any matching shape.
    end : int
        0 for the area at x = 0, 1 for the area at x = L.
    depth, half_width : float
        Area depth from the goal line and half width, metres (up to 1 m behind the goal line counts).

    Returns
    -------
    numpy.ndarray
        bool, same shape as `px`; False where the position is NaN.
    """
    x = px if end == 0 else L - px
    with np.errstate(invalid="ignore"):
        return (x >= -1.0) & (x <= depth) & (np.abs(py - W / 2) <= half_width)


def corner_formation(tracking: Tracking) -> np.ndarray:
    """A crowded penalty area (>= ~9 players of both teams) and a player at a corner flag of that end
    (Law 17). Live attacks fill the box too, but nobody stands at the flag then. Max over the two ends,
    smoothed over 1 s.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.

    Returns
    -------
    numpy.ndarray
        (n,) float32 score in [0, 1].
    """
    px, py = tracking.players[:, :, 0].astype(np.float64), tracking.players[:, :, 1].astype(np.float64)
    best = np.zeros(len(px), dtype=np.float64)
    for end in (0, 1):
        n_box = in_box(px, py, end, BOX_DEPTH, BOX_HALF_WIDTH).sum(1)
        flag_x = 0.0 if end == 0 else L
        with np.errstate(invalid="ignore"):
            d_flag = np.minimum(np.hypot(px - flag_x, py), np.hypot(px - flag_x, py - W))
        d_flag = np.where(np.isfinite(d_flag), d_flag, np.inf).min(1)
        score = sigmoid((n_box - 8.5) / 1.0) * sigmoid((CORNER_TAKER_M - d_flag) / 0.5)
        best = np.maximum(best, score)
    return rolling(best.astype(np.float32), FPS)


def wall_formation(tracking: Tracking, speed: np.ndarray, ball_ff: np.ndarray, ball_age: np.ndarray,
                   chunk: int = 20000, max_ball_age_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """A defensive wall: >= 3 team-mates standing still shoulder to shoulder (Law 13 free kick).

    Returns (wall_n, wall_ring): wall_n is 1 + the largest number of still team-mates within WALL_GAP_M
    of one still player (a line of 3 has a middle player with 2 such neighbours); pairs closer than
    DUPLICATE_M are one person tracked twice and do not count. wall_ring is the same restricted to
    players 7-11.5 m from the last-seen ball (9.15 m rule), which needs a ball seen within
    `max_ball_age_s`. "Still" means slower than 1.5 m/s; goalkeepers are excluded; both are the max over
    the two teams, smoothed over 1 s.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    speed : numpy.ndarray
        (n, n_slots) per-slot speed in m/s.
    ball_ff : numpy.ndarray
        (n, 2) last-seen ball position carried forward, metres.
    ball_age : numpy.ndarray
        (n,) seconds since that position was observed.
    chunk : int, optional
        Frames per chunk of the pairwise-distance computation (memory only, no effect on the result).
    max_ball_age_s : float, optional
        Oldest usable ball position for `wall_ring`, seconds.

    Returns
    -------
    wall_n, wall_ring : numpy.ndarray
        (n,) float32 each, head counts (0 where no player is still).
    """
    n = tracking.n_frames
    wall_n = np.zeros(n, dtype=np.float32)
    wall_ring = np.zeros(n, dtype=np.float32)
    for team_code in (0, 1):
        cols = np.flatnonzero((tracking.team == team_code) & ~tracking.goalkeeper)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            p = tracking.players[a:b][:, cols].astype(np.float32)
            s = speed[a:b][:, cols]
            d = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))
            # one person tracked by two slots: keep the first slot of each close pair only
            duplicate = np.triu(np.nan_to_num(d, nan=99.0) < DUPLICATE_M, k=1).any(1)
            still = np.isfinite(s) & (s < 1.5) & ~duplicate
            adj = (d < WALL_GAP_M) & still[:, :, None] & still[:, None, :]
            adj[:, np.arange(len(cols)), np.arange(len(cols))] = False
            deg = adj.sum(2)
            wall_n[a:b] = np.maximum(wall_n[a:b], np.where(still.any(1), 1 + deg.max(1), 0))
            bf = ball_ff[a:b].astype(np.float32)
            with np.errstate(invalid="ignore"):
                db = np.sqrt(((p - bf[:, None, :]) ** 2).sum(-1))
                ring = (db > WALL_RING[0]) & (db < WALL_RING[1]) & (ball_age[a:b] <= max_ball_age_s)[:, None]
            adj_r = adj & ring[:, :, None] & ring[:, None, :]
            deg_r = adj_r.sum(2)
            wall_ring[a:b] = np.maximum(wall_ring[a:b], np.where(ring.any(1), 1 + deg_r.max(1), 0))
    return rolling(wall_n, FPS), rolling(wall_ring, FPS)


def goal_kick_formation(tracking: Tracking, sign: np.ndarray, ball_ff: np.ndarray, ball_age: np.ndarray,
                        max_ball_age_s: float = 10.0) -> np.ndarray:
    """A goalkeeper inside his goal area, no opponent inside that penalty area (Law 16) and the last-seen
    ball in that goal area or beyond that goal line (a keeper alone waits in his box whenever play is at
    the other end, so the ball term is required).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    sign : numpy.ndarray
        (n,) +1 where the home team attacks +x, -1 otherwise.
    ball_ff : numpy.ndarray
        (n, 2) last-seen ball position carried forward, metres.
    ball_age : numpy.ndarray
        (n,) seconds since that position was observed.
    max_ball_age_s : float, optional
        Oldest usable ball position, seconds.

    Returns
    -------
    numpy.ndarray
        (n,) float32 score in [0, 1], max over the two teams, smoothed over 1 s.
    """
    px, py = tracking.players[:, :, 0].astype(np.float64), tracking.players[:, :, 1].astype(np.float64)
    bx, by = ball_ff[:, 0].astype(np.float64), ball_ff[:, 1].astype(np.float64)
    recent = ball_age <= max_ball_age_s
    best = np.zeros(len(px))
    for team_code, team_sign in ((0, 1.0), (1, -1.0)):
        gk = np.flatnonzero(tracking.goalkeeper & (tracking.team == team_code))
        if not len(gk):
            continue
        opponents = tracking.team != team_code
        defends_end0 = (sign * team_sign) > 0        # attacks +x, so defends x = 0
        for end in (0, 1):
            here = defends_end0 if end == 0 else ~defends_end0
            gk_in = in_box(px[:, gk], py[:, gk], end, SIX_DEPTH + 1.0, SIX_HALF_WIDTH + 1.0).any(1)
            opp_in = in_box(px[:, opponents], py[:, opponents], end, BOX_DEPTH, BOX_HALF_WIDTH).sum(1)
            x_ball = bx if end == 0 else L - bx
            with np.errstate(invalid="ignore"):
                ball_here = recent & (((x_ball < SIX_DEPTH + 2.0) & (np.abs(by - W / 2) < SIX_HALF_WIDTH + 2.0))
                                      | ((x_ball < 0.0) & (np.abs(by - W / 2) < BOX_HALF_WIDTH + 10.0)))
            score = here * gk_in * ball_here * sigmoid((0.5 - opp_in) / 0.3)
            best = np.maximum(best, score)
    return rolling(best.astype(np.float32), FPS)


def throw_in_formation(tracking: Tracking, speed: np.ndarray, ball_ff: np.ndarray, ball_age: np.ndarray) -> np.ndarray:
    """A nearly still player on or behind a touchline (Law 15) away from the corners, with >= 2 other
    players within 15 m, and the ball (when seen in the last second) within 3 m of him.

    The candidate thrower is the player furthest beyond a touchline at each frame.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    speed : numpy.ndarray
        (n, n_slots) per-slot speed in m/s.
    ball_ff : numpy.ndarray
        (n, 2) last-seen ball position carried forward, metres.
    ball_age : numpy.ndarray
        (n,) seconds since that position was observed.

    Returns
    -------
    numpy.ndarray
        (n,) float32 score in [0, 1], smoothed over 1 s.
    """
    px, py = tracking.players[:, :, 0].astype(np.float64), tracking.players[:, :, 1].astype(np.float64)
    with np.errstate(invalid="ignore"):
        behind = np.maximum(-py, py - W)                     # > 0 beyond a touchline
        along = (px > 3.0) & (px < L - 3.0)                  # corners are corner kicks
        cand = np.where(along & np.isfinite(behind), behind, -np.inf)
    j = np.argmax(cand, axis=1)
    rows = np.arange(len(px))
    behind_best = cand[rows, j]
    tx, ty = px[rows, j], py[rows, j]
    thrower_speed = np.nan_to_num(speed[rows, j], nan=0.0)
    with np.errstate(invalid="ignore"):
        d = np.hypot(px - tx[:, None], py - ty[:, None])
        near = ((d < 15.0) & (d > 0.5)).sum(1)
        d_ball = np.hypot(ball_ff[:, 0] - tx, ball_ff[:, 1] - ty)
    ball_ok = np.where((ball_age <= 1.0) & np.isfinite(d_ball), sigmoid((3.0 - d_ball) / 0.5), 1.0)
    score = (sigmoid((behind_best - THROW_BEHIND_M) / 0.15) * sigmoid((near - 1.5) / 0.5)
             * sigmoid((1.5 - thrower_speed) / 0.3) * ball_ok)
    score = np.nan_to_num(score, nan=0.0)
    return rolling(score.astype(np.float32), FPS)
