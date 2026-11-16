"""Team-level, slot-identity-free per-frame features: the input of the team model.

Track slots are fragments, not people (DATA.md), so nothing here follows a slot through time except
over a few frames (velocities). Every feature is an aggregate over a team, the ball or the scene, so
the same code runs on the challenge tracking and on the DFL matches.

Naming convention, which makes the home/away swap augmentation mechanical (see swap_teams):
    h_*   home-team feature        a_*   the same feature for the away team
    s_*   signed home-minus-away   -> negated under a swap
    xh_*  x coordinate in the home attacking frame   -> L - x under a swap
    yh_*  y coordinate in the home attacking frame   -> W - y under a swap
    anything else is team-symmetric and unchanged.
Rolling-context columns keep the prefix and add a suffix (add_context), windows in frames (25 = 1 s):
    __m25, __m75, __m250   centred rolling mean over 25 / 75 / 250 frames
    __delta                mean of the next 25 frames minus mean of the last 25
    __past62, __fut62      trailing mean of the last 62 frames / of the next 62 frames (2.5 s)
    __d62                  __fut62 minus __past62
e.g. "h_cx__m25" is the home centroid's 1 s centred mean. `med_spd_125` is likewise a 125-frame mean.

The team model reads the SHAPE view only (shape_columns): no ball column, so it judges possession from
how the two teams stand and move, which transfers from the DFL tracking to the challenge tracking.
"""

import numpy as np
import pandas as pd

from ..constants import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W
from ..io.tracking import Tracking
from .common import ffill_signed, forward_rolling, nan_stat, rolling, time_since, time_until

MAX_HUMAN_MPS = 12.0  # faster per-frame displacement is a slot switch, not a sprint

# Columns describing the challenge tracking rather than the game (how many slots are tracked, slot churn). They
# differ between providers, so they are dropped everywhere (portable()).
SENSOR_COLUMNS = ("h_n", "a_n", "n_tracked", "churn")

_BALL_PREFIXES = ("ball_", "xh_ball", "yh_ball", "h_d", "a_d", "h_n3", "a_n3", "h_n8", "a_n8",
                  "s_d1", "s_last_close", "s_next_close", "last_close_age", "next_close_in",
                  "s_close_frac", "h_gk_dball", "a_gk_dball")

CONTEXT_COLUMNS = ("h_cx", "a_cx", "s_cx", "h_vx", "a_vx", "s_vx", "h_len", "a_len",
                   "h_wid", "a_wid", "med_spd", "s_d1", "ball_vis", "ball_spd", "frac_slow")
# Team-shape columns that also get long, asymmetric context: comparing the 2.5 s before with the 2.5 s
# after, over a 10 s backdrop, separates "this team is attacking" from a momentary shape.
LONG_CONTEXT_COLUMNS = ("s_cx", "s_vx", "s_len", "s_wid", "h_cx", "a_cx", "h_vx", "a_vx", "h_sx", "a_sx",
                        "h_sy", "a_sy", "h_x90", "a_x90", "h_x10", "a_x10")


def team_shape_features(tracking: Tracking, sign: np.ndarray) -> tuple[dict, np.ndarray]:
    """Shape of each team in its own attacking frame: where it stands, how stretched, how fast.

    Outfield players only. Per team (prefix ``h_`` / ``a_``): tracked count ``n``, centroid ``cx`` /
    ``cy``, 10th / 90th x percentiles ``x10`` / ``x90``, length ``len`` and width ``wid`` (90th minus 10th
    percentile), spreads ``sx`` / ``sy``, 1 s mean forward velocity ``vx`` and 1 s median speed ``spd``;
    plus the signed differences ``s_cx``, ``s_vx``, ``s_len``, ``s_wid``.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    sign : numpy.ndarray
        (n,) +1 where the home team attacks +x, -1 otherwise.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        (n,) float32 arrays; positions in metres, velocities in m/s.
    speed : numpy.ndarray
        (n, n_slots) per-slot speed in m/s from frame differences; NaN on the first frame, where a
        position is missing, and above MAX_HUMAN_MPS (a slot switch, not a sprint).
    """
    px, py = tracking.players[:, :, 0], tracking.players[:, :, 1]
    vx_raw = np.vstack([np.full((1, px.shape[1]), np.nan, np.float32), np.diff(px, axis=0) * FPS])
    vy_raw = np.vstack([np.full((1, px.shape[1]), np.nan, np.float32), np.diff(py, axis=0) * FPS])
    speed = np.sqrt(vx_raw ** 2 + vy_raw ** 2)
    speed[speed > MAX_HUMAN_MPS] = np.nan
    out = {}
    outfield = ~tracking.goalkeeper
    for prefix, team_code, team_sign in (("h_", 0, 1.0), ("a_", 1, -1.0)):
        cols = (tracking.team == team_code) & outfield
        s = (sign * team_sign)[:, None]
        xa = np.where(s > 0, px[:, cols], L - px[:, cols])          # towards the opponent goal
        ya = np.where(s > 0, py[:, cols], W - py[:, cols])
        vxa = vx_raw[:, cols] * s
        vxa[~np.isfinite(speed[:, cols])] = np.nan
        out[prefix + "n"] = np.isfinite(xa).sum(1).astype(np.float32)
        out[prefix + "cx"] = nan_stat(np.nanmean, xa)
        out[prefix + "cy"] = nan_stat(np.nanmean, ya)
        out[prefix + "x10"] = nan_stat(lambda v, axis: np.nanpercentile(v, 10, axis=axis), xa)
        out[prefix + "x90"] = nan_stat(lambda v, axis: np.nanpercentile(v, 90, axis=axis), xa)
        out[prefix + "len"] = out[prefix + "x90"] - out[prefix + "x10"]
        out[prefix + "wid"] = (nan_stat(lambda v, axis: np.nanpercentile(v, 90, axis=axis), ya)
                               - nan_stat(lambda v, axis: np.nanpercentile(v, 10, axis=axis), ya))
        out[prefix + "sx"] = nan_stat(np.nanstd, xa)
        out[prefix + "sy"] = nan_stat(np.nanstd, ya)
        out[prefix + "vx"] = rolling(nan_stat(np.nanmean, vxa), 25)
        out[prefix + "spd"] = rolling(nan_stat(np.nanmedian, speed[:, cols]), 25)
    # Signed differences, each team in its own attacking frame, so a home/away swap negates them.
    out["s_cx"] = out["h_cx"] - out["a_cx"]
    out["s_vx"] = out["h_vx"] - out["a_vx"]
    out["s_len"] = out["h_len"] - out["a_len"]
    out["s_wid"] = out["h_wid"] - out["a_wid"]
    return out, speed


def ball_features(tracking: Tracking, sign: np.ndarray) -> tuple[dict, np.ndarray]:
    """Ball visibility, gaps, position, speed, kicks and out-of-pitch distance.

    Ball speed is a 5-frame central difference (the ATD ball is smoothed, single-frame differences are
    noise), capped at 45 m/s. Kick evidence is the change of the mean velocity vector between the 0.5 s
    before and after.

    Parameters
    ----------
    tracking : Tracking
        Ball positions in metres.
    sign : numpy.ndarray
        (n,) +1 where the home team attacks +x, -1 otherwise.

    Returns
    -------
    columns : dict of str to numpy.ndarray
        (n,) float32 arrays: ``ball_*`` (visibility, seconds since / until visible, gap length, speed
        in m/s, kick, out-of-pitch distance in metres, > 0 outside the lines) and ``xh_ball`` /
        ``yh_ball`` (position in the home attacking frame, metres).
    visible : numpy.ndarray
        (n,) bool, ball observed.
    """
    bx, by = tracking.ball[:, 0].astype(np.float64), tracking.ball[:, 1].astype(np.float64)
    visible = np.isfinite(bx)
    out = {"ball_vis": visible.astype(np.float32)}
    out["ball_since_vis"] = time_since(visible)
    out["ball_until_vis"] = time_until(visible)
    out["ball_gap"] = np.where(visible, 0.0, out["ball_since_vis"] + out["ball_until_vis"]).astype(np.float32)
    out["xh_ball"] = np.where(sign > 0, bx, L - bx).astype(np.float32)
    out["yh_ball"] = np.where(sign > 0, by, W - by).astype(np.float32)
    k = 2  # 5-frame central difference: the ATD ball is smoothed, single-frame differences are noise
    vx = np.full_like(bx, np.nan)
    vy = np.full_like(by, np.nan)
    vx[k:-k] = (bx[2 * k:] - bx[:-2 * k]) * FPS / (2 * k)
    vy[k:-k] = (by[2 * k:] - by[:-2 * k]) * FPS / (2 * k)
    speed = np.minimum(np.sqrt(vx ** 2 + vy ** 2), 45.0)  # a few glitch frames exceed any real shot
    out["ball_spd"] = speed.astype(np.float32)
    out["ball_spd_max25"] = rolling(np.nan_to_num(speed, nan=0.0), 25, "max")
    # Kick / deflection evidence: change of the velocity vector between the 0.5 s before and after.
    half = 12
    vx_before = rolling(vx, half, "mean", center=False)
    vy_before = rolling(vy, half, "mean", center=False)
    vx_after = np.roll(vx_before, -half)
    vy_after = np.roll(vy_before, -half)
    out["ball_kick"] = np.sqrt((vx_after - vx_before) ** 2 + (vy_after - vy_before) ** 2).astype(np.float32)
    out["ball_kick_max25"] = rolling(np.nan_to_num(out["ball_kick"], nan=0.0), 25, "max")
    outside = np.maximum.reduce([-bx, bx - L, -by, by - W])
    out["ball_out"] = outside.astype(np.float32)                   # > 0 means outside the lines
    out["ball_out_any25"] = rolling(np.nan_to_num(outside, nan=-99) > 0.5, 25, "max")
    out["ball_out_past"] = time_since(np.nan_to_num(outside, nan=-99) > 0.5)
    return out, visible


def proximity_features(tracking: Tracking, ball_visible: np.ndarray, sign: np.ndarray) -> dict:
    """Distances from the ball to each team, with a memory of which team was last in close control.

    "Close control": one team within 1.5 m of the ball and clearly (1 m) closer than the other.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    ball_visible : numpy.ndarray
        (n,) bool, ball observed (from `ball_features`).
    sign : numpy.ndarray
        (n,) +1 where the home team attacks +x, -1 otherwise.

    Returns
    -------
    dict of str to numpy.ndarray
        (n,) float32 arrays: per team the nearest and second-nearest distances ``d1`` / ``d2``
        (metres), players within 3 m / 8 m ``n3`` / ``n8``, 1 s past / next minimum of ``d1``, the
        goalkeeper's distance to the ball ``gk_dball`` and x in the team's attacking frame ``gk_x``;
        ``s_d1`` (> 0: home closer), the last / next close-control team ``s_last_close`` /
        ``s_next_close`` (+1 home, -1 away, 0 none yet), seconds since / until close control, and the
        signed close-control share over 1 s / 3 s.
    """
    px, py = tracking.players[:, :, 0], tracking.players[:, :, 1]
    bx, by = tracking.ball[:, 0], tracking.ball[:, 1]
    with np.errstate(all="ignore"):
        dist = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    out = {}
    nearest = {}
    for prefix, team_code in (("h_", 0), ("a_", 1)):
        d = np.where((tracking.team == team_code)[None, :], dist, np.nan)
        d_sorted = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
        d1 = np.where(np.isfinite(d_sorted[:, 0]), d_sorted[:, 0], np.nan).astype(np.float32)
        d2 = np.where(np.isfinite(d_sorted[:, 1]), d_sorted[:, 1], np.nan).astype(np.float32)
        nearest[prefix] = d1
        out[prefix + "d1"] = d1
        out[prefix + "d2"] = d2
        out[prefix + "n3"] = np.where(ball_visible, (d_sorted < 3.0).sum(1), np.nan).astype(np.float32)
        out[prefix + "n8"] = np.where(ball_visible, (d_sorted < 8.0).sum(1), np.nan).astype(np.float32)
        filled = np.nan_to_num(d1, nan=99.0)
        out[prefix + "d1_min_past25"] = rolling(filled, 25, "min", center=False)
        out[prefix + "d1_min_next25"] = forward_rolling(filled, 25, "min")
    h1, a1 = nearest["h_"], nearest["a_"]
    out["s_d1"] = (a1 - h1).astype(np.float32)          # > 0: home closer
    # "Close control": one team within 1.5 m of the ball and clearly (1 m) closer than the other.
    close_h = ball_visible & (h1 < 1.5) & (a1 - h1 > 1.0)
    close_a = ball_visible & (a1 < 1.5) & (h1 - a1 > 1.0)
    signal = np.where(close_h, 1.0, np.where(close_a, -1.0, 0.0))
    valid = close_h | close_a
    out["s_last_close"] = ffill_signed(signal, valid)
    out["s_next_close"] = ffill_signed(signal[::-1], valid[::-1])[::-1].copy()
    out["last_close_age"] = time_since(valid)
    out["next_close_in"] = time_until(valid)
    out["s_close_frac25"] = rolling(signal, 25)
    out["s_close_frac75"] = rolling(signal, 75)
    for prefix, team_code in (("h_", 0), ("a_", 1)):
        gk = np.flatnonzero(tracking.goalkeeper & (tracking.team == team_code))
        if len(gk):
            out[prefix + "gk_dball"] = nan_stat(np.nanmin, dist[:, gk])
            gx = tracking.players[:, gk[0], 0]
            team_sign = 1.0 if team_code == 0 else -1.0
            out[prefix + "gk_x"] = np.where(sign * team_sign > 0, gx, L - gx).astype(np.float32)
    return out


def scene_features(tracking: Tracking, speed: np.ndarray) -> dict:
    """Collective motion and sensor state that do not need the ball.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    speed : numpy.ndarray
        (n, n_slots) per-slot speed in m/s (from `team_shape_features`).

    Returns
    -------
    dict of str to numpy.ndarray
        (n,) float32 arrays: median speed over 1 s / 5 s ``med_spd`` / ``med_spd_125`` (m/s), share of
        players below 1 m/s ``frac_slow``, players more than 1 m outside the lines ``n_outside``, and the
        sensor columns ``churn`` (slot appearances / disappearances over ~4 s) and ``n_tracked``.
    """
    out = {}
    med = nan_stat(np.nanmedian, speed)
    out["med_spd"] = rolling(med, 25)
    out["med_spd_125"] = rolling(med, 125)
    with np.errstate(invalid="ignore"):
        out["frac_slow"] = rolling(nan_stat(np.nanmean, (speed < 1.0).astype(np.float32)
                                            + np.where(np.isfinite(speed), 0, np.nan)), 25)
    px, py = tracking.players[:, :, 0], tracking.players[:, :, 1]
    with np.errstate(invalid="ignore"):
        outside = (px < -1) | (px > L + 1) | (py < -1) | (py > W + 1)
    out["n_outside"] = rolling(outside.sum(1).astype(np.float32), 25)
    present = np.isfinite(px)
    changes = np.vstack([np.zeros((1, present.shape[1]), bool), present[1:] != present[:-1]]).sum(1)
    out["churn"] = rolling(changes.astype(np.float32), 101, "sum")
    out["n_tracked"] = present.sum(1).astype(np.float32)
    return out


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Centred 1 s / 3 s means and a 1 s future-minus-past delta for the key columns, plus a 10 s mean
    and 2.5 s past / future means and their difference for the team-shape columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-frame features containing CONTEXT_COLUMNS and LONG_CONTEXT_COLUMNS.

    Returns
    -------
    pandas.DataFrame
        `frame` with the context columns appended: ``<c>__m25``, ``<c>__m75``, ``<c>__delta`` for each
        context column and ``<c>__m250``, ``<c>__past62``, ``<c>__fut62``, ``<c>__d62`` for each long
        context column (the future windows wrap around at the end of the array).
    """
    extra = {}
    for column in CONTEXT_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float32)
        extra[column + "__m25"] = rolling(values, 25)
        extra[column + "__m75"] = rolling(values, 75)
        past = rolling(values, 25, center=False)
        extra[column + "__delta"] = (np.roll(past, -25) - past).astype(np.float32)
    for column in LONG_CONTEXT_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float32)
        extra[column + "__m250"] = rolling(values, 250)
        past = rolling(values, 62, center=False)
        future = np.roll(past, -62)
        extra[column + "__past62"] = past
        extra[column + "__fut62"] = future
        extra[column + "__d62"] = (future - past).astype(np.float32)
    return pd.concat([frame, pd.DataFrame(extra, index=frame.index)], axis=1)


def portable(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the sensor-specific columns (SENSOR_COLUMNS and their context variants).

    They describe the challenge tracking rather than the game and differ between providers.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-frame features.

    Returns
    -------
    pandas.DataFrame
        `frame` without the columns whose base name (before ``__``) is in SENSOR_COLUMNS.
    """
    drop = [c for c in frame.columns if c in SENSOR_COLUMNS or c.split("__")[0] in SENSOR_COLUMNS]
    return frame.drop(columns=drop)


def team_features(tracking: Tracking, sign: np.ndarray) -> pd.DataFrame:
    """The full per-frame feature table (float32, one row per frame), sensor columns dropped.

    Shape, ball, proximity and scene blocks, plus their rolling context (`add_context`).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres (the challenge tracking or a DFL match).
    sign : numpy.ndarray
        (n,) +1 where the home team attacks +x, -1 otherwise.

    Returns
    -------
    pandas.DataFrame
        (n_frames, n_columns) float32 features named by the module's prefix convention.
    """
    blocks = {}
    shape, speed = team_shape_features(tracking, sign)
    blocks.update(shape)
    ball, visible = ball_features(tracking, sign)
    blocks.update(ball)
    blocks.update(proximity_features(tracking, visible, sign))
    blocks.update(scene_features(tracking, speed))
    frame = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in blocks.items()})
    return portable(add_context(frame))


def is_shape_column(column: str) -> bool:
    """The shape view: no ball, Smart Tagging or camera column.

    Parameters
    ----------
    column : str
        A feature column name.

    Returns
    -------
    bool
        False if the name starts with a ball prefix.
    """
    return not column.startswith(_BALL_PREFIXES)


def shape_columns(frame: pd.DataFrame) -> list[str]:
    """The columns of the shape view (the team model's input), in table order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-frame features.

    Returns
    -------
    list of str
        The columns for which `is_shape_column` is True.
    """
    return [c for c in frame.columns if is_shape_column(c)]


def swap_teams(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frames seen with home and away exchanged (for symmetric augmentation).

    ``h_*`` and ``a_*`` columns are exchanged, ``s_*`` negated, ``xh_*`` mapped to L - x and ``yh_*`` to
    W - y; other columns are unchanged.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-frame features named by the module's prefix convention.

    Returns
    -------
    pandas.DataFrame
        Same index and column order as `frame` (every ``h_*`` column needs its ``a_*`` counterpart).
    """
    swapped = {}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if column.startswith("h_"):
            swapped["a_" + column[2:]] = values
        elif column.startswith("a_"):
            swapped["h_" + column[2:]] = values
        elif column.startswith("s_"):
            swapped[column] = -values
        elif column.startswith("xh_"):
            swapped[column] = L - values
        elif column.startswith("yh_"):
            swapped[column] = W - values
        else:
            swapped[column] = values
    return pd.DataFrame(swapped, index=frame.index)[frame.columns]
