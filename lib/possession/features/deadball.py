"""Dead-ball cues: one row of `db_*` columns per frame, and the hand-weighted cue rules built on them.

Stage 1 uses these cues twice: as input of the DFL-trained dead/alive head (the tag-free columns, see
tag_free_cues) and through the transparent cue rules (rule_terms + cue_rule_probability). Everything
is computed from tracking and Smart Tagging only; every constant is a physical or Laws-of-the-Game
scale documented next to it.

Groups:
    motion      db_spd_*, db_frac_*, db_decel                 collective slowdown / onset
    ball        db_ball_*, db_out_*, db_since_out_evidence, db_rest, ...  ball placed / out / missing
    formation   db_kickoff, db_corner, db_wall_*, db_goal_kick, db_throw_in
    events      db_onset, db_restart, db_since_onset, db_until_restart, db_since_restart
    tags        db_sp_*, db_goal_pending (Smart Tagging; all zero on DFL, which has no tags)

References: IFAB Laws of the Game 2023/24 (Laws 8, 13, 15, 16, 17); Siegle & Lames (2012), "Game
interruptions in elite soccer", J. Sports Sci. 30(7).
"""

import numpy as np
import pandas as pd

from ..constants import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W
from ..io.tracking import Tracking
from ..rules import formations, observed
from ..rules.kinematics import distance_outside, player_speeds
from .common import attack_sign_by_half, rolling, row_quantile, runs, sigmoid, time_since, time_until

CENTRE = (L / 2, W / 2)
WALK_MPS = 2.0            # walking pace; jogging is faster
STILL_MPS = 0.7           # standing / shuffling
BALL_STILL_MPS = 1.0      # a placed ball, above the noise of the smoothed track
BALL_STRUCK_MPS = 6.0     # a restart pass, kick or throw
GOAL_ACTIVE_MPS = 1.8     # 3 s median player speed of attacking play
SP_RESTART_SEARCH_S = 25.0  # a restart is taken within ~25 s of the SET PIECES window start
SP_QUIET_S = 1.0          # a set-piece window ends at resumed play only after 1 s without the ball played
SP_TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off")
WALK_P90_MPS, WALK_MED_MPS, WALK_MIN_S = 2.0, 1.3, 3.0

# Smart Tagging cue columns; the other cues exist on DFL too.
TAG_PREFIXES = ("db_sp_", "db_goal_pending")


def _past(values: np.ndarray, seconds: float, how: str = "mean") -> np.ndarray:
    """Statistic over the trailing window [t - seconds, t]."""
    return rolling(values, max(1, int(seconds * FPS)), how, center=False)


def _future(values: np.ndarray, seconds: float, how: str = "mean") -> np.ndarray:
    """Statistic over the leading window [t, t + seconds]."""
    window = max(1, int(seconds * FPS))
    return rolling(np.asarray(values)[::-1], window, how, center=False)[::-1].copy()


def _ffill_position(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last finite position carried forward, and its age in seconds (inf before the first)."""
    valid = np.isfinite(xy[:, 0])
    n = len(xy)
    index = np.where(valid, np.arange(n), -1)
    last = np.maximum.accumulate(index)
    out = np.where((last >= 0)[:, None], xy[np.maximum(last, 0)], np.nan).astype(np.float32)
    age = np.where(last >= 0, (np.arange(n) - last) / FPS, np.inf).astype(np.float32)
    return out, age


def collective_motion(speed: np.ndarray) -> dict:
    """Collective-motion cues: median and 90th-percentile player speed, walking and still shares.

    Stoppages make every player walk: the median and the fast tail of the speed distribution drop
    together. Trailing / leading 3 s means give the onset (deceleration) and restart (acceleration).

    Parameters
    ----------
    speed : numpy.ndarray
        (n, n_slots) per-slot player speed in m/s, NaN where unknown.

    Returns
    -------
    dict of str to numpy.ndarray
        (n,) float32 arrays: ``db_spd_med_{1,3,10}s``, ``db_spd_p90_{1,3}s`` (centred means, m/s),
        ``db_frac_walk_1s`` / ``db_frac_still_1s`` (share below WALK_MPS / STILL_MPS), the trailing /
        leading 3 s median ``db_spd_past3`` / ``db_spd_next3`` and ``db_decel`` = past - next
        (> 0: everyone is slowing down).
    """
    med = row_quantile(speed, 0.5)
    p90 = row_quantile(speed, 0.9)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(speed)
        walk = np.where(finite, speed < WALK_MPS, False).sum(1) / np.maximum(finite.sum(1), 1)
        still = np.where(finite, speed < STILL_MPS, False).sum(1) / np.maximum(finite.sum(1), 1)
    out = {
        "db_spd_med_1s": rolling(med, FPS),
        "db_spd_med_3s": rolling(med, 3 * FPS),
        "db_spd_med_10s": rolling(med, 10 * FPS),
        "db_spd_p90_1s": rolling(p90, FPS),
        "db_spd_p90_3s": rolling(p90, 3 * FPS),
        "db_frac_walk_1s": rolling(walk.astype(np.float32), FPS),
        "db_frac_still_1s": rolling(still.astype(np.float32), FPS),
    }
    past3, next3 = _past(med, 3), _future(med, 3)
    out["db_spd_past3"], out["db_spd_next3"] = past3, next3
    out["db_decel"] = (past3 - next3).astype(np.float32)     # > 0: everyone is slowing down
    return out


def ball_state(tracking: Tracking, med_speed_past: np.ndarray, rules: observed.OutRules) -> tuple[dict, np.ndarray]:
    """Visibility, smoothed speed, placed ball, observed out-of-play rules and gap length.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    med_speed_past : numpy.ndarray
        (n,) players' median speed in m/s (``db_spd_med_1s``), passed to the observed rules as the
        collective player speed.
    rules : observed.OutRules
        Ball-out refinements used by the ``db_out_*`` columns.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        (n,) float32 ``db_*`` arrays: ball visibility shares, speed (1 s median, 2 s max), the share of
        the last 2 s with a visible stationary ball, gap length and distance outside the lines, the
        observed rules (at rest, player outside, keeper holds, player carries, lost while slow) and
        the observed dead flags ``db_out_evidence`` (out | vanished at a line | at rest | player outside),
        ``db_out_until_restart``, ``db_ball_out_seen``, ``db_ball_vanished`` and ``db_since_out_evidence`` (seconds).
    ball_speed : numpy.ndarray
        (n,) float64 ball speed in m/s (5-frame central difference, capped at 45), NaN where unknown.
    """
    ball = tracking.ball.astype(np.float64)
    bx, by = ball[:, 0], ball[:, 1]
    visible = np.isfinite(bx)
    k = 2  # 5-frame central difference
    v = np.full(len(bx), np.nan)
    v[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) * FPS / (2 * k)
    v = np.minimum(v, 45.0)
    vis_f = visible.astype(np.float32)
    still = (visible & (np.nan_to_num(v, nan=99) < BALL_STILL_MPS)).astype(np.float32)
    out = {
        "db_ball_vis_1s": rolling(vis_f, FPS),
        "db_ball_vis_5s": rolling(vis_f, 5 * FPS),
        "db_ball_spd_1s": rolling(v, FPS, "median"),
        "db_ball_spd_max_2s": rolling(np.nan_to_num(v, nan=0.0), 2 * FPS, "max"),
        "db_ball_still_2s": _past(still, 2),          # share of the last 2 s with a visible, stationary ball
    }
    out["db_ball_gap_s"] = np.where(visible, 0.0, time_since(visible, 60) + time_until(visible, 60)).astype(np.float32)
    out["db_ball_out_m"] = np.nan_to_num(distance_outside(tracking.ball), nan=-5.0).astype(np.float32)
    out_mask = observed.ball_out_mask(tracking, graded_only=False, rules=rules)
    vanished = observed.ball_vanished_at_line(tracking, player_speed=med_speed_past)
    rest, outside = observed.ball_at_rest(tracking), observed.player_outside(tracking)
    out["db_rest"] = rest.astype(np.float32)
    out["db_player_outside"] = outside.astype(np.float32)
    out["db_keeper_holds"] = observed.ball_with_player(tracking, True).astype(np.float32)
    out["db_player_carries"] = observed.ball_with_player(tracking, False, reach_m=1.0).astype(np.float32)
    out["db_lost_slow"] = observed.ball_lost_while_slow(tracking, med_speed_past).astype(np.float32)
    out_evidence = out_mask | vanished | rest | outside
    out["db_out_until_restart"] = observed.observed_stoppages(tracking, player_speed=med_speed_past, rules=rules).astype(np.float32)
    out["db_ball_out_seen"] = out_mask.astype(np.float32)
    out["db_ball_vanished"] = vanished.astype(np.float32)
    out["db_out_evidence"] = out_evidence.astype(np.float32)
    out["db_since_out_evidence"] = time_since(out_evidence, 60)
    return out, v


def restart_spots(ball_ff: np.ndarray, age: np.ndarray, max_age_s: float = 5.0) -> dict:
    """Distances of the last-seen ball (carried forward up to `max_age_s`) to the corner arcs, the centre
    spot, the goal areas and the touchlines: where a restart would be taken.

    Parameters
    ----------
    ball_ff : numpy.ndarray
        (n, 2) last finite ball position carried forward, metres.
    age : numpy.ndarray
        (n,) seconds since that position was observed (inf before the first).
    max_age_s : float, optional
        Positions older than this give NaN distances.

    Returns
    -------
    dict of str to numpy.ndarray
        (n,) float32: ``db_ball_d_corner``, ``db_ball_d_centre``, ``db_ball_d_touch`` (metres, < 0
        outside), ``db_ball_in_six`` (0 / 1, goal area plus 1 m) and ``db_ball_age_s`` (capped at 60).
    """
    bx, by = ball_ff[:, 0].astype(np.float64), ball_ff[:, 1].astype(np.float64)
    ok = age <= max_age_s
    corners = np.array([[0, 0], [0, W], [L, 0], [L, W]], dtype=np.float64)
    d_corner = np.min(np.hypot(bx[:, None] - corners[None, :, 0], by[:, None] - corners[None, :, 1]), axis=1)
    d_centre = np.hypot(bx - CENTRE[0], by - CENTRE[1])
    in_six = (np.minimum(bx, L - bx) < formations.SIX_DEPTH + 1.0) & (np.abs(by - W / 2) < formations.SIX_HALF_WIDTH + 1.0)
    d_touch = np.minimum(by, W - by)   # < 0 outside
    nan = np.float32(np.nan)
    return {
        "db_ball_d_corner": np.where(ok, d_corner, nan).astype(np.float32),
        "db_ball_d_centre": np.where(ok, d_centre, nan).astype(np.float32),
        "db_ball_in_six": np.where(ok, in_six, nan).astype(np.float32),
        "db_ball_d_touch": np.where(ok, d_touch, nan).astype(np.float32),
        "db_ball_age_s": np.minimum(age, 60).astype(np.float32),
    }


def restart_score(ball_speed: np.ndarray, motion: dict, ball_visible: np.ndarray) -> np.ndarray:
    """Soft 'restart taken at this frame' score: a ball at rest (or missing) that starts to move now and
    is clearly struck within ~1.5 s, out of a frozen scene; a collective acceleration out of a frozen
    scene is a weaker, ball-free substitute.

    Parameters
    ----------
    ball_speed : numpy.ndarray
        (n,) ball speed in m/s, NaN where unknown.
    motion : dict
        Columns of `collective_motion` (``db_spd_past3``, ``db_decel`` are read).
    ball_visible : numpy.ndarray
        (n,) bool.

    Returns
    -------
    numpy.ndarray
        (n,) float32 score in [0, 1].
    """
    v = np.nan_to_num(ball_speed, nan=0.0)
    vis = ball_visible.astype(np.float32)
    lag = int(0.2 * FPS)
    pre = np.r_[np.zeros(lag), _past(v, 1.3, "max")[:-lag]]          # [t - 1.5 s, t - 0.2 s]
    pre_vis = np.r_[np.zeros(lag), _past(vis, 1.3)[:-lag]]
    post = _future(v, 1.5, "max")
    from_rest = sigmoid((2.5 - pre) / 0.4) * sigmoid((pre_vis - 0.3) / 0.1)
    from_missing = sigmoid((0.3 - pre_vis) / 0.1)
    moving_now = sigmoid((v - 2.5) / 0.4) * vis
    struck = moving_now * sigmoid((post - BALL_STRUCK_MPS) / 1.0) * np.maximum(from_rest, from_missing)
    frozen = sigmoid((1.4 - motion["db_spd_past3"]) / 0.2)
    accel = sigmoid((-motion["db_decel"] - 0.6) / 0.2)
    return (frozen * np.maximum(struck, 0.7 * accel)).astype(np.float32)


def onset_score(motion: dict, out_evidence: np.ndarray) -> np.ndarray:
    """Soft 'play just stopped' score: an observed ball-out onset, or a collective deceleration of more
    than ~0.8 m/s into walking pace (a whistle; tracking cannot hear it).

    Parameters
    ----------
    motion : dict
        Columns of `collective_motion` (``db_decel``, ``db_spd_next3`` are read).
    out_evidence : numpy.ndarray
        (n,) bool observed dead flag; its onsets count for 2 s (centred).

    Returns
    -------
    numpy.ndarray
        (n,) float32 score in [0, 1].
    """
    out_onset = np.r_[False, out_evidence[1:] & ~out_evidence[:-1]].astype(np.float32)
    out_onset = rolling(out_onset, 2 * FPS, "max")
    whistle = sigmoid((motion["db_decel"] - 0.8) / 0.2) * sigmoid((1.4 - motion["db_spd_next3"]) / 0.2)
    return np.maximum(out_onset, whistle).astype(np.float32)


def tag_cues(tags: pd.DataFrame, n: int, restart: np.ndarray, med_speed: np.ndarray,
             resumed: np.ndarray, played: np.ndarray) -> dict:
    """SET PIECES windows bracket a restart; GOALS ends at the following kick-off.

    Each SET PIECES window gives `db_sp_open` = 1 from its start to the first restart detected in
    [start, start + 25 s] (or to start + 8 s when none is detected), shortened to where play visibly
    resumed (R4f, after the ball has been out of play for SP_QUIET_S inside the window); plus
    `db_sp_since` / `db_sp_until_restart` and one column per set-piece type. GOALS: `db_goal_pending` = 1
    from the end of the scoring attack (last frame inside the interval with the players' 3 s median speed
    above GOAL_ACTIVE_MPS) to the kick-off restart.

    Parameters
    ----------
    tags : pandas.DataFrame
        Smart Tagging table (`io.tracking.load_smart_tagging`).
    n : int
        Number of frames.
    restart : numpy.ndarray
        (n,) restart score (`restart_score`); > 0.5 is a detected restart.
    med_speed : numpy.ndarray
        (n,) players' 3 s median speed in m/s.
    resumed : numpy.ndarray
        (n,) bool, onsets of visibly resumed play (``observed.play_resumed_onsets``).
    played : numpy.ndarray
        (n,) bool, ball being played (``observed.ball_played``).

    Returns
    -------
    dict of str to numpy.ndarray
        (n,) float32: ``db_sp_open``, ``db_sp_since`` / ``db_sp_until_restart`` (seconds, 60 where
        none), ``db_goal_pending`` and ``db_sp_<type>`` for each of SP_TYPES (1 while that type's window
        is open).
    """
    sp_open = np.zeros(n, np.float32)
    sp_since = np.full(n, 60.0, np.float32)
    sp_until = np.full(n, 60.0, np.float32)
    sp_type = {t: np.zeros(n, np.float32) for t in SP_TYPES}
    for row in tags[tags.code == "SET PIECES"].itertuples():
        a, b = max(int(row.start_frame), 0), min(int(row.end_frame), n)
        if b <= a:
            continue
        search = restart[a:min(n, a + int(SP_RESTART_SEARCH_S * FPS))] > 0.5
        hits = np.flatnonzero(search)
        stop = a + (int(hits[0]) if len(hits) else int(8 * FPS))
        quiet = np.flatnonzero(rolling((~played[a:a + len(search)]).astype(np.float32), int(SP_QUIET_S * FPS), "min") > 0.5)
        if len(quiet):
            late = np.zeros(len(search), dtype=bool)
            late[int(quiet[0]):] = True
            first = np.flatnonzero(resumed[a:a + len(search)] & late)
            if len(first):
                stop = min(stop, a + int(first[0]))       # it only ever shortens a window
        sp_open[a:stop] = 1.0
        sp_since[a:b] = np.minimum(sp_since[a:b], np.arange(b - a) / FPS)
        lo = max(0, stop - 60 * FPS)
        sp_until[lo:stop] = np.minimum(sp_until[lo:stop], (stop - np.arange(lo, stop)) / FPS)
        for t in SP_TYPES:
            if row.type.startswith(t):
                sp_type[t][a:stop] = 1.0
    goal_pending = np.zeros(n, np.float32)
    for row in tags[tags.code == "GOALS"].itertuples():
        a, b = max(int(row.start_frame), 0), min(int(row.end_frame), n)
        active = np.flatnonzero(med_speed[a:b] > GOAL_ACTIVE_MPS)
        goal = a + int(active[-1]) if len(active) else max(a, b - 90 * FPS)
        post = restart[max(0, b - 2 * FPS):min(n, b + 10 * FPS)]
        r = np.flatnonzero(post > 0.5)
        kick = max(0, b - 2 * FPS) + (int(r[0]) if len(r) else 2 * FPS)
        goal_pending[goal:kick] = 1.0
    out = {"db_sp_open": sp_open, "db_sp_since": sp_since, "db_sp_until_restart": sp_until,
           "db_goal_pending": goal_pending}
    for t, column in sp_type.items():
        out["db_sp_" + t.lower().replace(" ", "_").replace("-", "_")] = column
    return out


def dead_ball_cues(tracking: Tracking, tags: pd.DataFrame, rules: observed.OutRules) -> pd.DataFrame:
    """All dead-ball cues of the active match as a float32 DataFrame (`db_*` columns), one row per frame.

    Motion, ball, restart-spot, formation, event and tag groups (see the module docstring), plus
    ``db_walk`` (everyone walking for a whole run of >= 3 s).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    tags : pandas.DataFrame
        Smart Tagging table; empty on DFL (`io.tracking.empty_tags`), which zeroes the tag cues.
    rules : observed.OutRules
        Which ball-out refinements the cues' observed-rule columns (``db_out_*``, ``db_ball_out_seen``, ``db_ball_vanished``) use.

    Returns
    -------
    pandas.DataFrame
        (n_frames, n_cues) float32.
    """
    n = tracking.n_frames
    sign = attack_sign_by_half(n, tracking.home_attacks_positive_x_first_half)
    speed = player_speeds(tracking)
    cols = dict(collective_motion(speed))
    ball, ball_speed = ball_state(tracking, med_speed_past=cols["db_spd_med_1s"], rules=rules)
    cols.update(ball)
    visible = np.isfinite(tracking.ball[:, 0])
    ball_ff, age = _ffill_position(tracking.ball)
    cols.update(restart_spots(ball_ff, age))
    cols["db_kickoff"] = formations.kickoff_formation(tracking, sign)
    cols["db_corner"] = formations.corner_formation(tracking)
    cols["db_wall_n"], cols["db_wall_ring"] = formations.wall_formation(tracking, speed, ball_ff, age)
    cols["db_goal_kick"] = formations.goal_kick_formation(tracking, sign, ball_ff, age)
    cols["db_throw_in"] = formations.throw_in_formation(tracking, speed, ball_ff, age)
    restart = restart_score(ball_speed, cols, visible)
    onset = onset_score(cols, ball["db_out_evidence"] > 0.5)
    cols["db_restart"], cols["db_onset"] = restart, onset
    cols["db_since_onset"] = time_since(onset > 0.5, 60)
    cols["db_until_restart"] = time_until(restart > 0.5, 60)
    cols["db_since_restart"] = time_since(restart > 0.5, 60)
    cols.update(tag_cues(tags, n, restart, cols["db_spd_med_3s"], resumed=observed.play_resumed_onsets(tracking),
                         played=observed.ball_played(tracking.ball)))
    with np.errstate(invalid="ignore"):
        walk = (np.asarray(cols["db_spd_p90_1s"]) < WALK_P90_MPS) & (np.asarray(cols["db_spd_med_1s"]) < WALK_MED_MPS)
    walk_run = np.zeros(n, dtype=np.float32)
    for a, b in runs(walk):
        if b - a >= WALK_MIN_S * FPS:
            walk_run[a:b] = 1.0
    cols["db_walk"] = walk_run
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})


def tag_free_cues(cues: pd.DataFrame) -> pd.DataFrame:
    """The tag-free cues, which exist on DFL as well: the dead/alive head's cue input.

    Parameters
    ----------
    cues : pandas.DataFrame
        Output of `dead_ball_cues`.

    Returns
    -------
    pandas.DataFrame
        `cues` without the TAG_PREFIXES columns, with a fresh 0..n-1 index.
    """
    return cues[[c for c in cues.columns if not c.startswith(TAG_PREFIXES)]].reset_index(drop=True)


# -- the cue rules -------------------------------------------------------------------------------------

def rule_terms(cues: pd.DataFrame) -> pd.DataFrame:
    """The individual terms of the cue rules, each in [0, 1], one column per term.

    Parameters
    ----------
    cues : pandas.DataFrame
        Output of `dead_ball_cues` (tag columns included).

    Returns
    -------
    pandas.DataFrame
        (n_frames, 18) float32, NaN filled with 0. Columns, in this order: slow, fast, ball_out,
        since_out, kickoff, corner, wall, goal_kick, throw_in, ball_placed, sp_open, goal_pending,
        restart_passed, ball_moving, lost_slow, keeper_holds, player_carries, walk.
    """
    def g(column: str) -> np.ndarray:
        """One cue column as float64."""
        return cues[column].to_numpy(np.float64)

    slow = sigmoid((1.3 - g("db_spd_med_3s")) / 0.2) * sigmoid((3.0 - g("db_spd_p90_3s")) / 0.4)
    fast = sigmoid((g("db_spd_p90_1s") - 4.5) / 0.4)
    since_out = g("db_out_until_restart") * np.where(g("db_ball_vis_1s") > 0.5, 1.0, 0.0)   # only while the ball is seen
    placed = g("db_ball_still_2s") * sigmoid((1.6 - g("db_spd_med_3s")) / 0.2)
    wall = sigmoid((g("db_wall_n") - 2.5) / 0.3) * np.maximum(0.5, sigmoid((g("db_wall_ring") - 2.5) / 0.3))
    restart_passed = (g("db_since_restart") < 5).astype(np.float64)
    moving = sigmoid((np.nan_to_num(g("db_ball_spd_1s"), nan=0.0) - 4.0) / 1.0) * (g("db_ball_vis_1s") > 0.5)
    terms = {
        "slow": slow, "fast": fast, "ball_out": g("db_out_evidence"), "since_out": since_out,
        "kickoff": g("db_kickoff"), "corner": g("db_corner"), "wall": wall,
        "goal_kick": g("db_goal_kick"), "throw_in": g("db_throw_in"), "ball_placed": placed,
        "sp_open": g("db_sp_open"), "goal_pending": g("db_goal_pending"),
        "restart_passed": restart_passed, "ball_moving": moving,
        "lost_slow": g("db_lost_slow"), "keeper_holds": g("db_keeper_holds"),
        "player_carries": g("db_player_carries"), "walk": g("db_walk"),
    }
    return pd.DataFrame(terms).fillna(0.0).astype(np.float32)


def cue_rule_probability(terms: pd.DataFrame, weights: dict[str, float], bias: float) -> np.ndarray:
    """P(dead) of the cue rules: a logistic of the weighted terms (weights in nats, hand-set).

    Parameters
    ----------
    terms : pandas.DataFrame
        Output of `rule_terms`.
    weights : dict of str to float
        Log-odds weight of each term, in nats; summed in the dict's order.
    bias : float
        Intercept in nats.

    Returns
    -------
    numpy.ndarray
        (n_frames,) float64 P(dead) = sigmoid(bias + sum of weight * term).
    """
    z = np.full(len(terms), bias)
    for name, weight in weights.items():
        z += weight * terms[name].to_numpy(np.float64)
    return 1 / (1 + np.exp(-z))
