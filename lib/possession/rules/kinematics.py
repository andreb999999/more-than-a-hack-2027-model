"""Ball and player kinematics shared by the dead-ball rules and cues."""

import warnings

import numpy as np
import pandas as pd

from ..constants import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W
from ..features.common import row_quantile
from ..io.tracking import Tracking

MAX_HUMAN_MPS = 12.0   # faster per-frame displacement is a slot switch, not a sprint


def distance_outside(ball: np.ndarray) -> np.ndarray:
    """Metres beyond the nearest line (> 0 outside, < 0 inside, NaN when the ball is missing).

    Parameters
    ----------
    ball : numpy.ndarray
        (n, 2) positions in metres (corner origin).

    Returns
    -------
    numpy.ndarray
        (n,) float64 signed distance in metres.
    """
    bx, by = ball[:, 0].astype(np.float64), ball[:, 1].astype(np.float64)
    return np.maximum.reduce([-bx, bx - L, -by, by - W])


def ball_speed(ball: np.ndarray, k: int = 2) -> np.ndarray:
    """Ball speed (m/s) from a (2k+1)-frame central difference; the ATD ball is smoothed, so frame
    differences are mostly noise. NaN where the ball is missing.

    Parameters
    ----------
    ball : numpy.ndarray
        (n, 2) positions in metres.
    k : int, optional
        Half-width of the difference in frames.

    Returns
    -------
    numpy.ndarray
        (n,) float64 speed in m/s; NaN on the first and last `k` frames.
    """
    ball = ball.astype(np.float64)
    speed = np.full(len(ball), np.nan)
    step = ball[2 * k:] - ball[:-2 * k]
    speed[k:-k] = np.sqrt((step ** 2).sum(1)) * FPS / (2 * k)
    return speed


def player_speeds(tracking: Tracking) -> np.ndarray:
    """(n, slots) speed in m/s from frame differences; slot switches (> 12 m/s) are NaN.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.

    Returns
    -------
    numpy.ndarray
        (n_frames, n_slots) float32 m/s; NaN on the first frame and where a position is missing.
    """
    p = tracking.players.astype(np.float32)
    step = np.sqrt(np.diff(p[:, :, 0], axis=0) ** 2 + np.diff(p[:, :, 1], axis=0) ** 2) * FPS
    speed = np.vstack([np.full((1, p.shape[1]), np.nan, np.float32), step])
    speed[speed > MAX_HUMAN_MPS] = np.nan
    return speed


def fastest_players_speed(tracking: Tracking) -> np.ndarray:
    """90th percentile of the players' speeds per frame, 1 s rolling median: 'are the fastest players
    moving?'.

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.

    Returns
    -------
    numpy.ndarray
        (n_frames,) float64 m/s.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.Series(row_quantile(player_speeds(tracking), 0.9)).rolling(
            FPS, center=True, min_periods=1).median().to_numpy()


def keeper_distance(tracking: Tracking, frame: int) -> float:
    """Distance (m) from the ball to the nearest goalkeeper at `frame` (inf if either is missing).

    Parameters
    ----------
    tracking : Tracking
        Positions in metres.
    frame : int
        Video frame.

    Returns
    -------
    float
        Metres.
    """
    ball = tracking.ball[frame].astype(np.float64)
    keepers = tracking.players[frame].astype(np.float64)[np.asarray(tracking.goalkeeper, dtype=bool)]
    with np.errstate(invalid="ignore"):
        d = np.hypot(keepers[:, 0] - ball[0], keepers[:, 1] - ball[1])
    d = d[np.isfinite(d)]
    return float(d.min()) if len(d) else np.inf
