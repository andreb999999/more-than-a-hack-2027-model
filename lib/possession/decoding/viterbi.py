"""Explicit-duration Viterbi: per-frame H / A / D probabilities -> label sequence.

Each label is expanded into a left-to-right chain of `min_duration` sub-states, so a label, once
entered, lasts at least that many frames (a hidden semi-Markov model with a shifted-geometric
duration). Leaving a label costs `penalty_scale` times a relative switch cost. Emissions are scaled
likelihoods log P(k|x) - log P(k) (Bourlard & Morgan's hybrid trick). `dead_bias` is added to the D
log-emission. Each graded half is decoded on its own; ungraded frames get X.

The two free parameters, `penalty_scale` and `dead_bias`, can be chosen without any ground truth by
matching two priors: the number of H / A runs (with a neutral team split, the live stretches) and the
share of dead-ball time
(calibrate_penalty_scale, calibrate_dead_bias).
"""

import logging
from dataclasses import dataclass, replace

import numba
import numpy as np

from .. import timeline
from ..constants import A, D, H, X
from ..features.common import segment_bounds

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViterbiConfig:
    """Parameters of the explicit-duration Viterbi decoder.

    Attributes
    ----------
    min_duration : tuple of int
        (3,) minimum run length in frames for H, A, D (sub-states per label).
    switch : tuple of tuple of float
        (3, 3) relative switch costs ``switch[from][to]``.
    penalty_scale : float
        Nats per unit of switch cost: leaving label j for k costs ``penalty_scale * switch[j][k]``.
    prior : tuple of float
        (3,) class prior of the emissions, divided out (scaled likelihoods).
    floor : float
        Probability floor before taking logs.
    dead_bias : float
        Nats added to the D log-emission.
    max_sandwiched_alive : int
        Alive runs shorter than this between two D runs become D (0: off).
    """
    min_duration: tuple          # frames for H, A, D
    switch: tuple                # relative switch costs, [from][to]
    penalty_scale: float
    prior: tuple                 # class prior of the emissions
    floor: float                 # probability floor before taking logs
    dead_bias: float             # added to the D log-emission
    max_sandwiched_alive: int    # alive runs shorter than this between two D runs become D (0: off)


@numba.njit(cache=True)
def _viterbi_expanded(log_emit, min_dur, log_switch):
    """log_emit (T, K); min_dur (K,); log_switch (K, K) <= 0 off-diagonal. Returns labels (T,)."""
    T, K = log_emit.shape
    offsets = np.zeros(K + 1, dtype=np.int64)
    for k in range(K):
        offsets[k + 1] = offsets[k] + min_dur[k]
    S = offsets[K]
    label_of = np.zeros(S, dtype=np.int64)
    for k in range(K):
        for i in range(offsets[k], offsets[k + 1]):
            label_of[i] = k
    neg = -1e18
    score = np.full(S, neg)
    for k in range(K):
        score[offsets[k]] = log_emit[0, k]
    back = np.zeros((T, S), dtype=np.int32)
    new = np.empty(S)
    for t in range(1, T):
        for k in range(K):
            first, last = offsets[k], offsets[k + 1] - 1
            # entering label k at its first sub-state: from the last sub-state of any other label
            best, arg = neg, -1
            for j in range(K):
                if j == k:
                    continue
                cand = score[offsets[j + 1] - 1] + log_switch[j, k]
                if cand > best:
                    best, arg = cand, offsets[j + 1] - 1
            if min_dur[k] == 1:
                stay = score[first]
                if stay >= best:
                    best, arg = stay, first
                new[first] = best + log_emit[t, k]
                back[t, first] = arg
            else:
                new[first] = best + log_emit[t, k]
                back[t, first] = arg
                for i in range(first + 1, last):
                    new[i] = score[i - 1] + log_emit[t, k]
                    back[t, i] = i - 1
                # last sub-state: advance from the previous one or stay
                adv, stay = score[last - 1], score[last]
                if stay >= adv:
                    new[last] = stay + log_emit[t, k]
                    back[t, last] = last
                else:
                    new[last] = adv + log_emit[t, k]
                    back[t, last] = last - 1
        for i in range(S):
            score[i] = new[i]
    # must end in a completed (last) sub-state when possible
    best, state = neg, 0
    for k in range(K):
        i = offsets[k + 1] - 1
        if score[i] > best:
            best, state = score[i], i
    if best <= neg / 2:
        for i in range(S):
            if score[i] > best:
                best, state = score[i], i
    out = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        out[t] = label_of[state]
        state = back[t, state]
    return out


def log_emissions(proba: np.ndarray, config: ViterbiConfig) -> np.ndarray:
    """Scaled log-likelihoods log P(k|x) - log P(k), with `dead_bias` added to D.

    Parameters
    ----------
    proba : numpy.ndarray
        (T, 3) H / A / D probabilities; clipped to [floor, 1] and renormalised per row.
    config : ViterbiConfig
        Uses `floor`, `prior` and `dead_bias`.

    Returns
    -------
    numpy.ndarray
        (T, 3) float64 log-emissions in nats.
    """
    proba = np.clip(np.asarray(proba, dtype=np.float64), config.floor, 1.0)
    proba = proba / proba.sum(1, keepdims=True)
    out = np.log(proba) - np.log(np.asarray(config.prior))[None, :]
    out[:, D] += config.dead_bias
    return out


def decode_block(proba: np.ndarray, config: ViterbiConfig) -> np.ndarray:
    """Decode one contiguous block (no half split, no sandwich rule).

    Parameters
    ----------
    proba : numpy.ndarray
        (T, 3) H / A / D probabilities.
    config : ViterbiConfig
        Decoder parameters.

    Returns
    -------
    numpy.ndarray
        (T,) int64 label codes H / A / D; the path ends in a completed run when one is reachable.
    """
    log_switch = -config.penalty_scale * np.asarray(config.switch, dtype=np.float64)
    return _viterbi_expanded(log_emissions(proba, config), np.asarray(config.min_duration, dtype=np.int64), log_switch)


def remove_sandwiched_alive(block: np.ndarray, max_len: int) -> np.ndarray:
    """Turn H/A runs shorter than `max_len` that sit between two D runs into D (a short 'possession'
    inside a stoppage is treated as a false restart).

    Parameters
    ----------
    block : numpy.ndarray
        (T,) label codes.
    max_len : int
        Runs strictly shorter than this many frames are filled; <= 0 disables the rule.

    Returns
    -------
    numpy.ndarray
        (T,) a copy with the sandwiched runs set to D (`block` itself when the rule is off or the
        block is empty). Any non-D run counts as alive, X included.
    """
    if max_len <= 0 or len(block) == 0:
        return block
    out = block.copy()
    starts, stops = segment_bounds(out)
    codes = out[starts]
    for i in range(1, len(starts) - 1):
        if codes[i] != D and codes[i - 1] == D and codes[i + 1] == D and stops[i] - starts[i] < max_len:
            out[starts[i]:stops[i]] = D
    return out


def decode(proba: np.ndarray, config: ViterbiConfig) -> np.ndarray:
    """Decode each graded half separately; ungraded frames get X. Returns label codes.

    Each half is decoded with `decode_block`, then `remove_sandwiched_alive` with
    ``config.max_sandwiched_alive``.

    Parameters
    ----------
    proba : numpy.ndarray
        (n_frames, 3) H / A / D probabilities of the whole match.
    config : ViterbiConfig
        Decoder parameters.

    Returns
    -------
    numpy.ndarray
        (n_frames,) int8 codes H / A / D, X outside the graded halves of the active timeline.
    """
    labels = np.full(len(proba), X, dtype=np.int8)
    for start, stop in timeline.halves():
        if stop <= start:
            continue
        block = decode_block(proba[start:stop], config)
        labels[start:stop] = remove_sandwiched_alive(block, config.max_sandwiched_alive)
    return labels


def possession_count(labels: np.ndarray) -> int:
    """Number of H / A runs inside the graded halves of the active timeline.

    Parameters
    ----------
    labels : numpy.ndarray
        (n_frames,) label codes.

    Returns
    -------
    int
        Possessions (runs of equal H or A labels), counted per half.
    """
    count = 0
    for start, stop in timeline.halves():
        block = labels[start:stop]
        if stop <= start:
            continue
        alive = (block == H) | (block == A)
        starts, _ = segment_bounds(block)
        count += int(alive[starts].sum())
    return count


def dead_share(labels: np.ndarray) -> float:
    """Share of match-play frames labelled D (pre-kick-off frames excluded).

    Parameters
    ----------
    labels : numpy.ndarray
        (n_frames,) label codes.

    Returns
    -------
    float
        In [0, 1]; needs the kick-offs of the active timeline (``timeline.match_mask``).
    """
    match = timeline.match_mask(len(labels))   # pre-kick-off frames are D regardless: match play only
    return float((labels[match] == D).mean())


def _bisect(evaluate, lo: float, hi: float, target: float, increasing: bool, geometric: bool, iters: int,
            tol: float) -> tuple[float, float]:
    """x in [lo, hi] with evaluate(x) ~ target for a monotone evaluate. Returns (x, value)."""
    history = []
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi)) if geometric else 0.5 * (lo + hi)
        value = evaluate(mid)
        history.append((mid, value))
        if (value < target) == increasing:
            lo = mid
        else:
            hi = mid
        if abs(value - target) <= tol:
            break
    best = min(history, key=lambda item: abs(item[1] - target))
    return best[0], best[1]


def calibrate_penalty_scale(proba: np.ndarray, config: ViterbiConfig, target_live_stretches: int, low: float,
                            high: float, iters: int, tol: float) -> ViterbiConfig:
    """Geometric bisection on penalty_scale so the decoded match has ~target possessions (a prior).

    The possession count decreases with `penalty_scale`. The bisection stops after `iters` steps or
    when within `tol`; the best value seen is kept.

    Parameters
    ----------
    proba : numpy.ndarray
        (n_frames, 3) H / A / D probabilities.
    config : ViterbiConfig
        Starting configuration (all other fields kept).
    target_live_stretches : int
        Target number of H / A runs (`possession_count`).
    low, high : float
        Bisection interval of `penalty_scale`.
    iters : int
        Maximum bisection steps.
    tol : float
        Stop when the count is this close to the target.

    Returns
    -------
    ViterbiConfig
        `config` with the calibrated `penalty_scale`.
    """
    scale, count = _bisect(lambda x: possession_count(decode(proba, replace(config, penalty_scale=x))),
                           low, high, target_live_stretches, increasing=False, geometric=True, iters=iters, tol=tol)
    if abs(count - target_live_stretches) > 0.05 * target_live_stretches:
        log.info("switch penalty at %.3g: %d H / A runs (bound %d)", scale, count, target_live_stretches)
    return replace(config, penalty_scale=scale)


def calibrate_dead_bias(proba: np.ndarray, config: ViterbiConfig, target_dead_share: float, low: float,
                        high: float, iters: int, tol: float) -> ViterbiConfig:
    """Linear bisection on dead_bias so that the D share of match play is ~target (a prior).

    The D share increases with `dead_bias`. The bisection stops after `iters` steps or when within
    `tol`; the best value seen is kept, and a miss by more than 0.03 is logged as a warning.

    Parameters
    ----------
    proba : numpy.ndarray
        (n_frames, 3) H / A / D probabilities.
    config : ViterbiConfig
        Starting configuration (all other fields kept).
    target_dead_share : float
        Target share of match-play frames decoded as D (`dead_share`).
    low, high : float
        Bisection interval of `dead_bias`, nats.
    iters : int
        Maximum bisection steps.
    tol : float
        Stop when the share is this close to the target.

    Returns
    -------
    ViterbiConfig
        `config` with the calibrated `dead_bias`.
    """
    bias, share = _bisect(lambda x: dead_share(decode(proba, replace(config, dead_bias=x))),
                          low, high, target_dead_share, increasing=True, geometric=False, iters=iters, tol=tol)
    if abs(share - target_dead_share) > 0.03:
        log.warning("dead-ball calibration missed: share %.3f for target %.3f", share, target_dead_share)
    return replace(config, dead_bias=bias)
