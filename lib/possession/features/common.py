"""Small array helpers shared by the feature and rule modules."""

import warnings

import numpy as np
import pandas as pd

from ..constants import FPS
from .. import timeline


def rolling(values: np.ndarray, window: int, how: str = "mean", center: bool = True) -> np.ndarray:
    """Rolling statistic (pandas semantics, min_periods=1) as float32.

    Parameters
    ----------
    values : numpy.ndarray
        (n,) input series; NaN is skipped as in pandas.
    window : int
        Window length in frames.
    how : str, optional
        Name of the pandas ``Rolling`` method ("mean", "min", "max", "sum", "median", ...).
    center : bool, optional
        Centred window if True, else trailing (frames t - window + 1 .. t).

    Returns
    -------
    numpy.ndarray
        (n,) float32.
    """
    roller = pd.Series(values).rolling(window, center=center, min_periods=1)
    return getattr(roller, how)().to_numpy(dtype=np.float32)


def forward_rolling(values: np.ndarray, window: int, how: str = "min") -> np.ndarray:
    """Statistic over [t, t + window).

    Parameters
    ----------
    values : numpy.ndarray
        (n,) input series.
    window : int
        Window length in frames.
    how : str, optional
        Name of the pandas ``Rolling`` method.

    Returns
    -------
    numpy.ndarray
        (n,) float32; near the end the window is truncated (min_periods=1).
    """
    return rolling(np.asarray(values)[::-1], window, how, center=False)[::-1]


def time_since(event: np.ndarray, cap_s: float = 30.0) -> np.ndarray:
    """Seconds since the last frame where `event` was True (`cap_s` where never, capped).

    Parameters
    ----------
    event : numpy.ndarray
        (n,) bool.
    cap_s : float, optional
        Cap in seconds, also used before the first event.

    Returns
    -------
    numpy.ndarray
        (n,) float32 seconds; 0 on event frames.
    """
    n = len(event)
    index = np.where(event, np.arange(n), -1)
    last = np.maximum.accumulate(index)
    out = (np.arange(n) - last) / FPS
    out[last < 0] = cap_s
    return np.minimum(out, cap_s).astype(np.float32)


def time_until(event: np.ndarray, cap_s: float = 30.0) -> np.ndarray:
    """Seconds until the next frame where `event` is True (capped).

    Parameters
    ----------
    event : numpy.ndarray
        (n,) bool.
    cap_s : float, optional
        Cap in seconds, also used after the last event.

    Returns
    -------
    numpy.ndarray
        (n,) float32 seconds; 0 on event frames.
    """
    return time_since(np.asarray(event)[::-1], cap_s)[::-1].copy()


def ffill_signed(signal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Carry the last valid value forward (0 before the first).

    Parameters
    ----------
    signal : numpy.ndarray
        (n,) values.
    valid : numpy.ndarray
        (n,) bool, frames whose value is kept and carried forward.

    Returns
    -------
    numpy.ndarray
        (n,) float32.
    """
    n = len(signal)
    index = np.where(valid, np.arange(n), -1)
    last = np.maximum.accumulate(index)
    return np.where(last >= 0, signal[np.maximum(last, 0)], 0).astype(np.float32)


def run_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, stops) of every True run of a boolean mask, half-open.

    Parameters
    ----------
    mask : numpy.ndarray
        (n,) bool.

    Returns
    -------
    starts, stops : numpy.ndarray
        (n_runs,) int64 each.
    """
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0]))
    return edges[::2], edges[1::2]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open (start, stop) of every True run of a boolean mask.

    Parameters
    ----------
    mask : numpy.ndarray
        (n,) bool.

    Returns
    -------
    list of tuple
        ``[(start, stop), ...]`` in order.
    """
    return list(zip(*run_bounds(mask)))


def segment_bounds(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, stops) of every run of equal values in a label sequence, half-open.

    Parameters
    ----------
    labels : numpy.ndarray
        (n,) label codes, n >= 1.

    Returns
    -------
    starts, stops : numpy.ndarray
        (n_runs,) int each; the runs tile [0, n).
    """
    if len(labels) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.r_[0, changes], np.r_[changes, len(labels)]


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, stop, value) of every run of equal values in a label sequence.

    Parameters
    ----------
    labels : numpy.ndarray
        (n,) label codes, n >= 1.

    Returns
    -------
    list of tuple
        ``[(start, stop, value), ...]``, half-open, in order.
    """
    starts, stops = segment_bounds(labels)
    return list(zip(starts, stops, labels[starts]))


def attack_sign_by_half(n: int, home_attacks_positive_x_first_half: bool = True) -> np.ndarray:
    """+1 where the home team attacks towards +x, -1 otherwise, per frame of the active match.

    Teams swap ends at half-time; the switch is placed midway between the two halves.

    Parameters
    ----------
    n : int
        Number of frames.
    home_attacks_positive_x_first_half : bool, optional
        Attack direction of the home team in the first half.

    Returns
    -------
    numpy.ndarray
        (n,) float32, +1 or -1.
    """
    first = 1.0 if home_attacks_positive_x_first_half else -1.0
    sign = np.full(n, first, dtype=np.float32)
    (_, first_stop), (second_start, _) = timeline.halves()
    sign[(first_stop + second_start) // 2:] = -first
    return sign


def nan_stat(fn, values: np.ndarray, axis: int = 1) -> np.ndarray:
    """A NaN-aware reduction with all-NaN warnings silenced, as float32.

    Parameters
    ----------
    fn : callable
        Reduction called as ``fn(values, axis=axis)``, e.g. ``numpy.nanmean``.
    values : numpy.ndarray
        Input array, e.g. (n_frames, n_slots).
    axis : int, optional
        Axis to reduce.

    Returns
    -------
    numpy.ndarray
        The reduction as float32; NaN where a slice is all NaN.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(values, axis=axis).astype(np.float32)


def row_quantile(values: np.ndarray, q: float) -> np.ndarray:
    """Per-row quantile ignoring NaN (vectorised: NaN sorts last, index by the finite count).

    The quantile is the lower order statistic ``floor(q * (count - 1))``, no interpolation.

    Parameters
    ----------
    values : numpy.ndarray
        (n, m) values, NaN where missing.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    numpy.ndarray
        (n,) float32; NaN for rows without a finite value.
    """
    s = np.sort(values, axis=1)
    count = np.isfinite(values).sum(1)
    idx = np.clip(np.floor(q * (count - 1)).astype(np.int64), 0, values.shape[1] - 1)
    out = s[np.arange(len(s)), idx].astype(np.float32)
    out[count == 0] = np.nan
    return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, with `z` clipped to [-50, 50].

    Parameters
    ----------
    z : numpy.ndarray
        Log-odds.

    Returns
    -------
    numpy.ndarray
        Probabilities, same shape as `z`.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
