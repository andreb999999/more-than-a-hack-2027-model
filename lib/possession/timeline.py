"""The timeline of the match currently being processed: its graded halves and kick-off frames.

Every frame-indexed rule (forced dead-ball rules, decoder, playbook) works half by half and needs to
know where the graded halves of the match are. Most of the time that is the challenge match. While a DFL
training table is built, the same code runs on a DFL match, whose halves are laid end to end on its
own frame axis. Rather than passing the halves through every function, the active timeline is set
once with a context manager:

    with timeline.use(Timeline(halves=game.halves, kickoffs=(h1_start, h2_start))):
        ...                         # everything inside reads the DFL match's halves

Outside any `use` block the timeline is the challenge match with its kick-offs not yet known; the pipeline
detects them from the data (rules.observed.detect_kickoffs) and then enters
`use(Timeline(GRADED_HALVES, kickoffs))`.
"""

import contextlib
from dataclasses import dataclass

import numpy as np

from .constants import GRADED_HALVES


@dataclass(frozen=True)
class Timeline:
    """The graded halves and kick-off frames of one match.

    Attributes
    ----------
    halves : tuple
        ``((start, stop), (start, stop))``: the two graded halves as half-open video-frame ranges.
    kickoffs : tuple or None
        ``(first, second)``: kick-off frame of each half; None until detected.
    """
    halves: tuple                 # ((start, stop), (start, stop)), half-open video frames
    kickoffs: tuple | None = None  # kick-off frame of each half; None until detected


_ACTIVE = [Timeline(halves=GRADED_HALVES)]


def active() -> Timeline:
    """The active match timeline.

    Returns
    -------
    Timeline
        The innermost `use` timeline, or the challenge match (kick-offs unknown) outside any `use` block.
    """
    return _ACTIVE[-1]


def halves() -> tuple:
    """The graded halves of the active match.

    Returns
    -------
    tuple
        ``((start, stop), (start, stop))``, half-open video-frame ranges.
    """
    return active().halves


def kickoffs() -> tuple:
    """The kick-off frames of the active match.

    Returns
    -------
    tuple of int
        Kick-off frame of the first and of the second half.

    Raises
    ------
    RuntimeError
        If the active timeline has no kick-offs (outside a `use` block that sets them).
    """
    found = active().kickoffs
    if found is None:
        raise RuntimeError("kick-offs of this match are not known yet: enter timeline.use(...) with them")
    return found


@contextlib.contextmanager
def use(timeline: Timeline):
    """Make `timeline` the active match timeline inside the block.

    Blocks nest; the previous timeline is restored on exit, also on an exception.

    Parameters
    ----------
    timeline : Timeline
        The timeline to activate; its frames are converted to plain ``int``.

    Yields
    ------
    None
    """
    halves_ = tuple((int(a), int(b)) for a, b in timeline.halves)
    kicks = None if timeline.kickoffs is None else tuple(int(k) for k in timeline.kickoffs)
    _ACTIVE.append(Timeline(halves=halves_, kickoffs=kicks))
    try:
        yield
    finally:
        _ACTIVE.pop()


def graded_mask(n: int) -> np.ndarray:
    """Boolean mask of the graded frames of the active match.

    Parameters
    ----------
    n : int
        Number of frames of the match.

    Returns
    -------
    numpy.ndarray
        (n,) bool, True inside the graded halves.
    """
    mask = np.zeros(n, dtype=bool)
    for start, stop in halves():
        mask[start:stop] = True
    return mask


def match_mask(n: int) -> np.ndarray:
    """Graded frames of actual match play: each graded half from its kick-off on.

    Parameters
    ----------
    n : int
        Number of frames of the match.

    Returns
    -------
    numpy.ndarray
        (n,) bool, True from each kick-off to the end of its graded half.

    Raises
    ------
    RuntimeError
        If the kick-offs of the active match are not known (see `kickoffs`).
    """
    mask = graded_mask(n)
    for (start, _), kickoff in zip(halves(), kickoffs()):
        mask[start:kickoff] = False
    return mask
