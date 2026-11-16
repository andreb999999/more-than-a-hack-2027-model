"""Post-decoding rules that follow from how the metric scores a submission.

    overshoot_dead_edges  possessions are extended ~1 s into each neighbouring stoppage: under temporal
                          IoU an undershoot of x frames costs x/L, an overshoot only x/(L + x), so erring
                          long is cheaper.
    free_frames           every frame outside play is D (before each kick-off, half-time, after the final
                          whistle; the scorer ignores the ungraded ones), and the team that kicks off holds
                          the ball from the kick-off frame.
"""

import numpy as np

from .. import timeline
from ..constants import A, D, H
from ..features.common import segments


def overshoot_dead_edges(labels: np.ndarray, frames: int, min_dead: int = 25) -> np.ndarray:
    """Shrink every D run by `frames` at each end that touches a possession, keeping >= `min_dead`.

    The freed frames take the label of the neighbouring possession. The cut per side is limited so that
    at least `min_dead` frames of D remain (split evenly when both ends touch a possession).

    Parameters
    ----------
    labels : numpy.ndarray
        (n,) label codes.
    frames : int
        Overshoot per side, frames.
    min_dead : int, optional
        Shortest D run left, frames (runs already shorter are not cut).

    Returns
    -------
    numpy.ndarray
        (n,) a copy; only the graded halves of the active timeline are changed.
    """
    out = labels.copy()
    for start, stop in timeline.halves():
        if stop <= start:
            continue
        block = out[start:stop]
        parts = segments(block)
        for i, (a, b, code) in enumerate(parts):
            if code != D:
                continue
            left = parts[i - 1][2] if i > 0 else None
            right = parts[i + 1][2] if i + 1 < len(parts) else None
            grow_left, grow_right = left in (H, A), right in (H, A)
            sides = int(grow_left) + int(grow_right)
            if not sides:
                continue
            per_side = min(frames, max(0, (b - a) - min_dead) // sides)
            cut_left = per_side if grow_left else 0
            cut_right = per_side if grow_right else 0
            if cut_left:
                block[a:a + cut_left] = left
            if cut_right:
                block[b - cut_right:b] = right
        out[start:stop] = block
    return out


def free_frames(labels: np.ndarray) -> np.ndarray:
    """D outside play (before each detected kick-off and after each half); from each kick-off frame the
    ball belongs to the team that kicked off.

    "The team that kicked off" is read from the labels: a D run starting at the kick-off frame takes
    the label of the first H / A frame after it.

    Parameters
    ----------
    labels : numpy.ndarray
        (n,) label codes.

    Returns
    -------
    numpy.ndarray
        (n,) a copy; every frame outside [kick-off, end of half) is D, ungraded frames included.
        Needs the kick-offs of the active timeline.
    """
    out = labels.copy()
    kickoffs = timeline.kickoffs()
    play = np.zeros(len(out), dtype=bool)
    for (_, stop), kickoff in zip(timeline.halves(), kickoffs):
        play[kickoff:stop] = True
    out[~play] = D
    for kickoff in kickoffs:
        if out[kickoff] != D:
            continue
        stop = kickoff
        while stop < len(out) and out[stop] == D:
            stop += 1
        if stop < len(out) and out[stop] in (H, A):
            out[kickoff:stop] = out[stop]   # the team that kicked off keeps the ball
    return out


def apply_playbook(labels: np.ndarray, overshoot_frames: int) -> np.ndarray:
    """`free_frames`, then (if `overshoot_frames`) `overshoot_dead_edges` and `free_frames` again.

    Parameters
    ----------
    labels : numpy.ndarray
        (n,) decoded label codes.
    overshoot_frames : int
        Overshoot per side, frames; 0 disables it.

    Returns
    -------
    numpy.ndarray
        (n,) label codes.
    """
    out = free_frames(labels)
    if overshoot_frames:
        out = overshoot_dead_edges(out, frames=overshoot_frames)
        out = free_frames(out)
    return out
