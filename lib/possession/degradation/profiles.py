"""Statistics of the challenge tracking file, measured from the file alone (no labels).

A few sensor constants of SpeedBinProfile (ball smoothing, ball lead, minimum visible run, speed bins)
are fixed defaults rather than measured.

The DFL tracking is clean; the challenge tracking is not (the ball is often missing and smoothed, player
tracks are fragmented into anonymous slots). Models trained on DFL only transfer if the DFL matches are
first made to share these statistics (simulate.py).

    SpeedBinProfile   ball gaps by collective-speed bin, smoothing, lead, slot-run lengths.
                      Used for the team model's training data.
    GameStateProfile  ball gaps in play and in stoppages and by distance to the lines, the
                      out-of-pitch share in stoppages, the stoppage ball's speeds. Takes a dead/alive
                      state per frame (pipeline/training_data.py: stoppage_state).
                      Used for the dead/alive head's training data.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .. import timeline
from ..constants import FPS
from ..features.common import rolling, run_bounds
from ..io.tracking import Tracking
from ..rules.kinematics import ball_speed, distance_outside


@dataclass
class SpeedBinProfile:
    """Sensor statistics of the challenge match, by collective-speed bin.

    `measure_speed_bin_profile` fills the measured fields; `speed_edges`, `ball_sigma`, `ball_lead` and
    `min_visible_run` keep their defaults.

    Attributes
    ----------
    missing_share : tuple of float
        P(ball missing | speed bin), one per bin.
    gap_lengths : tuple of tuple of int
        Empirical ball-gap lengths (frames) per speed bin, binned by the gap's first frame.
    slot_run_frames : tuple of int
        Empirical lengths (frames) of the runs in which a slot is tracked.
    presence : float
        Mean share of the 10 outfielders per team that are tracked.
    speed_edges : tuple of float
        Bin edges (m/s) of the 5 s median player speed.
    ball_sigma : float
        Gaussian smoothing of the ball, frames.
    ball_lead : int
        Frames by which the challenge ball runs ahead of the players.
    min_visible_run : int
        Shortest visible run of the challenge ball, frames.
    """
    missing_share: tuple
    gap_lengths: tuple
    slot_run_frames: tuple
    presence: float
    speed_edges: tuple = (0.0, 0.8, 1.1, 1.5, 99.0)
    ball_sigma: float = 6.0
    ball_lead: int = 4
    min_visible_run: int = 15


LINE_EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, 999.0)   # distance to the nearest line; beyond a line counts as 0
ONSET_JITTER = 25          # frames: window around a stoppage's start in which its ball loss starts
TAIL_RANGE = (0, 60)       # frames: after the restart the ball is found again within this range, uniform


@dataclass
class GameStateProfile:
    """The challenge tracking by game state, measured on match-play frames.

    Attributes
    ----------
    play_missing_share, stop_missing_share : float
        Share of play / stoppage frames with the ball missing.
    play_gap_lengths, stop_gap_lengths : tuple of int
        Empirical ball-gap lengths (frames) of gaps starting in play / in a stoppage.
    play_line_hazard : tuple of float
        Relative gap-start hazard in play by distance-to-nearest-line bin (LINE_EDGES).
    stop_beyond_visible : float
        Share of visible stoppage frames with the ball > 0.25 m beyond a line.
    stop_onset_missing : float
        Share of stoppages with the ball missing within +-1 s of their start.
    play_onset_bins : tuple of float
        Share of the challenge play gap starts per distance-to-line bin.
    stop_ball_speeds : tuple of float
        A sample (every 5th) of the challenge tracking's visible stoppage ball speeds, m/s.
    stop_still_share : float
        Share of the challenge tracking's visible stoppage frames with the ball slower than 0.5 m/s.
    base : SpeedBinProfile or None
        The speed-bin profile: smoothing, lead, minimum run, player fragmentation.
    """
    play_missing_share: float
    stop_missing_share: float
    play_gap_lengths: tuple
    stop_gap_lengths: tuple
    play_line_hazard: tuple          # relative gap-start hazard in play by distance-to-nearest-line bin
    stop_beyond_visible: float       # share of visible stoppage frames with the ball > 0.25 m beyond a line
    stop_onset_missing: float        # share of stoppages with the ball missing within +-1 s of their start
    play_onset_bins: tuple = ()      # share of the challenge play gap starts per distance-to-line bin
    stop_ball_speeds: tuple = ()     # the challenge tracking's visible stoppage ball speeds (m/s), a sample
    stop_still_share: float = 0.05   # share of the challenge tracking's visible stoppage frames with the ball slower than 0.5 m/s
    base: SpeedBinProfile | None = None   # the speed-bin profile: smoothing, lead, minimum run, player fragmentation


def median_speed_125(players: np.ndarray) -> np.ndarray:
    """Median player speed per frame, 5 s rolling mean (slot switches > 12 m/s ignored).

    Parameters
    ----------
    players : numpy.ndarray
        (n, n_slots, 2) positions in metres, NaN where missing.

    Returns
    -------
    numpy.ndarray
        (n,) float32 m/s (125-frame centred mean); the first frame repeats the second.
    """
    v = np.diff(players, axis=0) * FPS
    speed = np.sqrt((v ** 2).sum(-1))
    speed[speed > 12] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(speed, axis=1)          # keeps the input precision (float64 inside degrade)
    med = np.r_[med[:1], med]
    return rolling(med, 125)


def line_bin(distance_inside: np.ndarray) -> np.ndarray:
    """Distance-to-nearest-line bin index (LINE_EDGES).

    Parameters
    ----------
    distance_inside : numpy.ndarray
        Metres inside the nearest line (negative beyond it, counted as 0; NaN counted as 50 m).

    Returns
    -------
    numpy.ndarray
        int bin indices in 0 .. len(LINE_EDGES) - 2, same shape as the input.
    """
    d = np.clip(np.nan_to_num(distance_inside, nan=50.0), 0.0, None)
    return np.clip(np.digitize(d, LINE_EDGES) - 1, 0, len(LINE_EDGES) - 2)


def measure_speed_bin_profile(tracking: Tracking) -> SpeedBinProfile:
    """The challenge tracking's ball-gap and slot statistics on the graded frames, by collective-speed bin.

    Parameters
    ----------
    tracking : Tracking
        The challenge match; the active timeline gives the graded frames.

    Returns
    -------
    SpeedBinProfile
        Measured `missing_share`, `gap_lengths` (a bin without gaps gets the overall median length),
        `slot_run_frames` and `presence`; the other fields keep their defaults.
    """
    graded = timeline.graded_mask(tracking.n_frames)
    speed = median_speed_125(tracking.players)
    missing = ~np.isfinite(tracking.ball[:, 0])
    edges = SpeedBinProfile.speed_edges
    bins = np.clip(np.digitize(speed, edges) - 1, 0, len(edges) - 2)
    share, lengths = [], []
    starts, stops = run_bounds(missing & graded)
    for b in range(len(edges) - 1):
        in_bin = graded & (bins == b)
        share.append(float(missing[in_bin].mean()) if in_bin.any() else 0.0)
        lens = [int(e - s) for s, e in zip(starts, stops) if bins[s] == b]
        lengths.append(tuple(lens) if lens else (int(np.median(stops - starts)),))
    present = np.isfinite(tracking.players[:, :, 0]) & graded[:, None]
    slot_runs = []
    for slot in range(present.shape[1]):
        s, e = run_bounds(present[:, slot])
        slot_runs += list(e - s)
    outfield = []
    for t in (0, 1):
        cols = (tracking.team == t) & ~tracking.goalkeeper
        outfield.append(np.isfinite(tracking.players[graded][:, cols, 0]).sum(1).mean() / 10.0)
    return SpeedBinProfile(missing_share=tuple(share), gap_lengths=tuple(lengths),
                          slot_run_frames=tuple(int(r) for r in slot_runs),
                          presence=float(min(1.0, np.mean(outfield))))


def measure_game_state_profile(tracking: Tracking, dead: np.ndarray, base: SpeedBinProfile) -> GameStateProfile:
    """GameStateProfile on the match-play frames of the challenge tracking, and a dead/alive state per frame.

    Must run inside a timeline with known kick-offs (match play = graded frames from each kick-off).

    Parameters
    ----------
    tracking : Tracking
        The challenge match.
    dead : numpy.ndarray
        (n_frames,) bool dead-ball state (`pipeline.training_data.stoppage_state`).
    base : SpeedBinProfile
        The speed-bin profile, stored as `base`.

    Returns
    -------
    GameStateProfile
    """
    graded = timeline.match_mask(tracking.n_frames)
    missing = ~np.isfinite(tracking.ball[:, 0])
    play, stop = graded & ~dead, graded & dead
    gaps = [(a, b) for a, b in zip(*run_bounds(missing & graded))]
    play_gaps = [b - a for a, b in gaps if play[a]]
    stop_gaps = [b - a for a, b in gaps if stop[a]]
    with np.errstate(invalid="ignore"):
        to_line = -distance_outside(tracking.ball)
    onset_bins = np.array([line_bin(to_line[a - 1]) for a, _ in gaps if a > 0 and play[a] and np.isfinite(to_line[a - 1])])
    vis = play & ~missing & np.isfinite(to_line)
    vis_bins = line_bin(to_line[vis])
    hazard = []
    for k in range(len(LINE_EDGES) - 1):
        share_onset = (onset_bins == k).mean() if len(onset_bins) else 0.0
        share_vis = (vis_bins == k).mean() if len(vis_bins) else 0.0
        hazard.append(float(share_onset / share_vis) if share_vis > 0 else 1.0)
    with np.errstate(invalid="ignore"):
        beyond = distance_outside(tracking.ball) > 0.25
    stops = [(a, b) for a, b in zip(*run_bounds(stop))]
    onset = [missing[max(a - ONSET_JITTER, 0):a + ONSET_JITTER].any() for a, _ in stops]
    spd = ball_speed(tracking.ball)
    vis_stop = spd[stop & np.isfinite(spd)]
    return GameStateProfile(play_missing_share=float(missing[play].mean()), stop_missing_share=float(missing[stop].mean()),
                            play_gap_lengths=tuple(play_gaps), stop_gap_lengths=tuple(stop_gaps),
                            play_line_hazard=tuple(hazard), stop_beyond_visible=float(beyond[stop & ~missing].mean()),
                            stop_onset_missing=float(np.mean(onset)) if onset else 0.5,
                            play_onset_bins=tuple(float((onset_bins == k).mean()) for k in range(len(LINE_EDGES) - 1)),
                            stop_ball_speeds=tuple(np.round(vis_stop[::5], 2).tolist()),
                            stop_still_share=float((vis_stop < 0.5).mean()) if len(vis_stop) else 0.05,
                            base=base)
