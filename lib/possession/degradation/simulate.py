"""Make clean DFL tracking look like the challenge tracking (deterministic for a given seed).

    degrade_by_speed_bin    ball gaps by collective-speed bin, ball smoothing and lead, minimum visible run,
                            player-track fragmentation. Team model training data.
    degrade_by_game_state   the same defects by game state: the ball lost at stoppage onsets and found
                            again after the restart, gaps starting near the lines in play, out-of-pitch
                            balls hidden in stoppages; DFL's frozen stoppage ball is first replaced by a
                            synthetic handled-ball path (stoppage_ball), or "ball frozen = dead" would be
                            a giveaway. Dead/alive head training data.
"""

import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..constants import FPS, PITCH_LENGTH, PITCH_WIDTH
from ..features.common import run_bounds, runs
from ..io.tracking import Tracking
from ..rules.kinematics import distance_outside
from .profiles import ONSET_JITTER, TAIL_RANGE, SpeedBinProfile, GameStateProfile, line_bin, median_speed_125

PLACED_S = (1.0, 3.0)        # the ball is still (placed / held) for 1-3 s before the restart


def _smooth_lead_fragment(ball: np.ndarray, missing: np.ndarray, players: np.ndarray, base: SpeedBinProfile,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Shared tail of both degradations: Gaussian smoothing within each visible run, the ball lead, the
    minimum visible run, and the player fragmentation (present runs drawn from the challenge slot-run lengths,
    absences sized so the tracked share equals the challenge tracking's). Returns (ball, players)."""
    n = len(ball)
    smooth = np.full_like(ball, np.nan)
    for s, e in zip(*run_bounds(~missing)):
        segment = ball[s:e]
        if np.isfinite(segment).all() and base.ball_sigma > 0:
            smooth[s:e] = gaussian_filter1d(segment, base.ball_sigma, axis=0, mode="nearest")
        else:
            smooth[s:e] = segment
    if base.ball_lead:
        smooth = np.vstack([smooth[base.ball_lead:], np.full((base.ball_lead, 2), np.nan)])
    for s, e in zip(*run_bounds(np.isfinite(smooth[:, 0]))):
        if e - s < base.min_visible_run:
            smooth[s:e] = np.nan
    slot_runs = np.asarray(base.slot_run_frames, dtype=np.float64)
    mean_run = float(slot_runs.mean())
    mean_gap = mean_run * (1 - base.presence) / max(base.presence, 1e-3)
    for slot in range(players.shape[1]):
        t = int(rng.integers(0, int(mean_run)))
        while t < n:
            gap = max(1, int(rng.exponential(mean_gap))) if mean_gap > 0 else 0
            players[t:t + gap, slot] = np.nan
            t += gap + int(slot_runs[rng.integers(len(slot_runs))])
    return smooth, players


def degrade_by_speed_bin(tracking: Tracking, profile: SpeedBinProfile, seed: int) -> Tracking:
    """A copy of clean tracking with the challenge tracking's defects.

    1. Ball gaps: starts drawn with a per-frame hazard that depends on the 5 s median player speed bin,
       lengths drawn from the challenge gaps that start in the same bin, so the missing share per bin matches the challenge tracking
       (only the share still lacking in each bin is added).
    2-3. Smoothing, lead, minimum visible run and player fragmentation (_smooth_lead_fragment).

    Parameters
    ----------
    tracking : Tracking
        Clean tracking (a DFL match); not modified.
    profile : SpeedBinProfile
        The challenge tracking's speed-bin statistics.
    seed : int
        Seed of the random generator; the result is deterministic for a given seed.

    Returns
    -------
    Tracking
        A new Tracking with float32 degraded players and ball; team, goalkeepers and attack direction
        are shared with the input.
    """
    rng = np.random.default_rng(seed)
    n = tracking.n_frames
    players = tracking.players.astype(np.float64).copy()
    ball = tracking.ball.astype(np.float64).copy()
    speed = median_speed_125(players)
    bins = np.clip(np.digitize(speed, profile.speed_edges) - 1, 0, len(profile.speed_edges) - 2)
    missing = ~np.isfinite(ball[:, 0])
    hazard = np.zeros(len(profile.missing_share))
    for b, (share, lengths) in enumerate(zip(profile.missing_share, profile.gap_lengths)):
        in_bin = bins == b
        already = float(missing[in_bin].mean()) if in_bin.any() else 0.0
        add = min(0.95, max(0.0, (share - already) / max(1e-6, 1.0 - already)))
        hazard[b] = add / ((1 - add) * max(1.0, float(np.mean(lengths))))
    draws = rng.random(n)
    t = 0
    while t < n:
        if draws[t] < hazard[bins[t]]:
            lengths = profile.gap_lengths[bins[t]]
            length = int(lengths[rng.integers(len(lengths))])
            missing[t:t + length] = True
            t += length
        t += 1
    smooth, players = _smooth_lead_fragment(ball, missing, players, profile, rng)
    return Tracking(players=players.astype(np.float32), ball=smooth.astype(np.float32),
                    team=tracking.team, goalkeeper=tracking.goalkeeper,
                    home_attacks_positive_x_first_half=tracking.home_attacks_positive_x_first_half)


def _leg(p0: np.ndarray, p1: np.ndarray, frames: int, v_leg: float, rng: np.random.Generator,
         ar: float = 0.9, sd: float = 0.4) -> np.ndarray:
    """Positions from p0 to p1 over `frames`, mean speed v_leg along the line plus AR(1) velocity noise,
    corrected linearly so the leg ends exactly on p1."""
    if frames <= 1:
        return np.repeat(p1[None, :], max(frames, 0), axis=0)
    direction = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-6)
    noise = np.zeros((frames, 2))
    for t in range(1, frames):
        noise[t] = ar * noise[t - 1] + rng.normal(0, sd, 2)
    velocity = direction[None, :] * v_leg + noise                 # m/s
    path = p0 + np.cumsum(velocity / FPS, axis=0)
    drift = path[-1] - p1
    return path - drift[None, :] * np.linspace(0, 1, frames)[:, None]


def stoppage_ball(tracking: Tracking, dead: np.ndarray, prof: GameStateProfile, seed: int) -> Tracking:
    """A copy whose ball, inside each stoppage, follows a synthetic handled-ball path instead of DFL's
    frozen position: legs with speeds drawn from the challenge visible stoppage ball speeds and durations of
    0.8-3 s, heading towards the restart spot (random angle, sd 60 degrees), with still dwells in between
    so the still share matches the challenge tracking; the last leg ends on the restart position, then held 1-3 s (placed ball).

    Only leg speeds of at least 0.5 m/s are drawn. A stoppage is left as it is when it touches the
    first or last frame, or when the ball is not seen on the frames just before and just after it. A few
    cm of Gaussian jitter is added so the ball is never exactly still.

    Parameters
    ----------
    tracking : Tracking
        Clean tracking (a DFL match); not modified.
    dead : numpy.ndarray
        (n_frames,) bool, the match's own dead-ball labels.
    prof : GameStateProfile
        The challenge tracking's game-state statistics (`stop_ball_speeds`, `stop_still_share`).
    seed : int
        Seed; the generator is seeded with ``seed + 1``.

    Returns
    -------
    Tracking
        A shallow copy whose `ball` is replaced (same dtype); players are shared with the input.
    """
    speeds = np.asarray(prof.stop_ball_speeds, dtype=np.float64)
    moving = speeds[speeds >= 0.5] if (speeds >= 0.5).any() else np.array([1.8])
    still = float(np.clip(prof.stop_still_share, 0.0, 0.9))
    rng = np.random.default_rng(seed + 1)
    ball = tracking.ball.astype(np.float64).copy()
    lo, hi = np.array([-3.0, -3.0]), np.array([PITCH_LENGTH + 3, PITCH_WIDTH + 3])
    for a, b in runs(np.asarray(dead, dtype=bool)):
        if a == 0 or b >= len(ball):
            continue
        p0, p1 = ball[a - 1], ball[b]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        n = b - a
        placed = min(int(rng.uniform(*PLACED_S) * FPS), n // 2)
        budget = n - placed
        move_budget = int(budget * (1 - still))
        parts, pos, used = [], p0.copy(), 0
        while used < move_budget - 2 * FPS:
            v = float(rng.choice(moving))
            frames = min(int(rng.uniform(0.8, 3.0) * FPS), move_budget - used - FPS)
            to_goal = p1 - pos
            angle = np.arctan2(to_goal[1], to_goal[0]) + rng.normal(0, np.pi / 3)
            nxt = np.clip(pos + v * frames / FPS * np.array([np.cos(angle), np.sin(angle)]), lo, hi)
            parts.append(("move", pos.copy(), nxt, frames, v))
            pos, used = nxt, used + frames
        last = max(FPS, move_budget - used)
        parts.append(("move", pos.copy(), p1, last, np.linalg.norm(p1 - pos) / (last / FPS)))
        dwell_total = max(0, budget - sum(x[3] for x in parts))
        dwell = (rng.dirichlet(np.ones(len(parts) + 1)) * dwell_total).astype(int)
        path = []
        for i, (_, q0, q1, frames, v) in enumerate(parts):
            path.append(np.repeat(q0[None, :], dwell[i], axis=0))
            path.append(_leg(q0, q1, frames, v, rng))
        path.append(np.repeat(p1[None, :], max(0, n - sum(len(x) for x in path)), axis=0))
        path = np.concatenate(path)[:n]
        if len(path) < n:
            path = np.vstack([path, np.repeat(p1[None, :], n - len(path), axis=0)])
        ball[a:b] = path + rng.normal(0, 0.02, path.shape)     # a few cm of jitter: never exactly still
    out = copy.copy(tracking)
    out.ball = ball.astype(tracking.ball.dtype)
    return out


def degrade_by_game_state(tracking: Tracking, profile: GameStateProfile, dead: np.ndarray, seed: int) -> Tracking:
    """The challenge tracking's defects by game state (`dead`: the source match's own dead/alive labels).

    1. Stoppage onset: in a share of the stoppages (the profile's stop_onset_missing) the ball is lost from a frame
       within +-1 s of the stoppage's start until 0-2.4 s after its restart.
    2. Other ball gaps: per-frame start hazards for play and stoppage, lengths drawn from the challenge gaps; in play
       scaled by the distance to the nearest line; a gap started in a stoppage ends 0-2.4 s after the restart.
    3. In stoppages, visible balls beyond a line are hidden, whole stretches at a time, until the visible
       out-of-pitch share matches the challenge tracking.
    The hazards are calibrated by simulation (the same random draws each pass) so the final missing shares
    equal the profile's; then smoothing, lead, minimum visible run and player fragmentation as in the speed-bin degradation.

    Parameters
    ----------
    tracking : Tracking
        Tracking to degrade (normally the output of `stoppage_ball`); not modified.
    profile : GameStateProfile
        The challenge tracking's game-state statistics; ``profile.base`` supplies the speed-bin smoothing, lead, minimum run and
        fragmentation.
    dead : numpy.ndarray
        (n_frames,) bool, the source match's own dead-ball labels.
    seed : int
        Seed of the random generator; the result is deterministic for a given seed.

    Returns
    -------
    Tracking
        A new Tracking with float32 degraded players and ball; team, goalkeepers and attack direction
        are shared with the input.
    """
    base = profile.base
    rng = np.random.default_rng(seed)
    n = tracking.n_frames
    players = tracking.players.astype(np.float64).copy()
    ball = tracking.ball.astype(np.float64).copy()
    dead = np.asarray(dead, dtype=bool)
    with np.errstate(invalid="ignore"):
        to_line = -distance_outside(ball)
        beyond = (distance_outside(ball) > 0.25) & dead
    bins = line_bin(to_line)
    base_missing = ~np.isfinite(ball[:, 0])
    stops = [(int(a), int(b)) for a, b in zip(*run_bounds(dead))]
    stop_end = np.zeros(n, dtype=np.int64)
    for a, b in stops:
        stop_end[a:b] = b
    def tail(u: float) -> int:
        """Frames after the restart until the ball is found again, for a uniform draw u."""
        return int(TAIL_RANGE[0] + u * (TAIL_RANGE[1] - TAIL_RANGE[0]))

    onset_u, onset_j, onset_t = rng.random(len(stops)), rng.integers(-ONSET_JITTER, ONSET_JITTER + 1, len(stops)), rng.random(len(stops))
    draws, picks, tails_u = rng.random(n), rng.random(n), rng.random(n)
    beyond_runs = [(int(a), int(b)) for a, b in zip(*run_bounds(beyond))]
    order = rng.permutation(len(beyond_runs))
    beyond_delay = rng.integers(0, ONSET_JITTER + 1, len(beyond_runs))   # the ball is lost 0-1 s after it crosses
    rel = np.asarray(profile.play_line_hazard, dtype=np.float64)
    weights = np.bincount(bins[~dead], minlength=len(rel)) / max((~dead).sum(), 1)
    rel = rel / max(float((rel * weights).sum()), 1e-9)

    hazard_starts = []

    def simulate(hp: float, hs: float, p_onset: float) -> np.ndarray:
        miss = base_missing.copy()
        hazard_starts.clear()
        for k, (a, b) in enumerate(stops):                     # 1. loss at the stoppage onset
            if onset_u[k] < p_onset:
                miss[max(a + int(onset_j[k]), 0):min(b + tail(onset_t[k]), n)] = True
        t = 0                                                   # 2. hazard gaps
        while t < n:
            if dead[t]:
                hit, pool = draws[t] < hs, profile.stop_gap_lengths
            else:
                hit, pool = draws[t] < hp * rel[bins[t]], profile.play_gap_lengths
            if hit:
                if not dead[t] and not miss[t]:
                    hazard_starts.append(t)
                length = int(pool[int(picks[t] * len(pool))])
                if dead[t]:
                    length = min(length, int(stop_end[t] - t) + tail(tails_u[t]))
                miss[t:t + length] = True
                t += length
            t += 1
        vis_stop = dead & ~miss                                  # 3. out-of-pitch ball in stoppages
        if vis_stop.any():
            for k in order:
                if (beyond & ~miss).sum() / max((dead & ~miss).sum(), 1) <= profile.stop_beyond_visible:
                    break
                a, b = beyond_runs[k]
                miss[min(a + int(beyond_delay[k]), b):b] = True
        return miss

    def odds(x: float) -> float:
        return x / max(1 - x, 1e-6)

    h_play = profile.play_missing_share / ((1 - profile.play_missing_share) * float(np.mean(profile.play_gap_lengths)))
    h_stop = profile.stop_missing_share / ((1 - profile.stop_missing_share) * float(np.mean(profile.stop_gap_lengths)))
    target_bins = np.asarray(profile.play_onset_bins, dtype=np.float64) if len(profile.play_onset_bins) else None
    p_onset = profile.stop_onset_missing
    def damp(r: np.ndarray | float) -> np.ndarray | float:   # damped multiplicative updates: the passes interact
        return np.sqrt(r)

    for _ in range(12):
        miss = simulate(h_play, h_stop, p_onset)
        got_play = float(miss[~dead].mean()) if (~dead).any() else profile.play_missing_share
        got_stop = float(miss[dead].mean()) if dead.any() else profile.stop_missing_share
        h_play *= damp(odds(profile.play_missing_share) / max(odds(got_play), 1e-6))
        h_stop = max(h_stop * damp(odds(profile.stop_missing_share) / max(odds(got_stop), 1e-6)), 0.0)
        if stops:                         # share of stoppages that lose the ball within +-1 s of their start
            got_onset = np.mean([miss[max(a - ONSET_JITTER, 0):a + ONSET_JITTER].any() for a, _ in stops])
            p_onset = float(np.clip(p_onset + 0.5 * (profile.stop_onset_missing - got_onset), 0.0, 1.0))
        if target_bins is not None and hazard_starts:   # only the play gap starts the line hazard creates
            starts = np.asarray(hazard_starts)
            starts = starts[starts > 0]
            got = np.bincount(bins[starts - 1], minlength=len(rel)) / max(len(starts), 1)
            rel *= np.where(got > 0, np.clip(damp(target_bins / np.maximum(got, 1e-6)), 0.5, 2.0), 1.0)
            rel /= max(float((rel * weights).sum()), 1e-9)
    missing = simulate(h_play, h_stop, p_onset)
    ball[missing] = np.nan
    smooth, players = _smooth_lead_fragment(ball, missing, players, base, rng)
    return Tracking(players=players.astype(np.float32), ball=smooth.astype(np.float32),
                    team=tracking.team, goalkeeper=tracking.goalkeeper,
                    home_attacks_positive_x_first_half=tracking.home_attacks_positive_x_first_half)


def degrade_with_stoppage_ball(tracking: Tracking, profile: GameStateProfile, dead: np.ndarray, seed: int) -> Tracking:
    """The full game-state degradation of a DFL match: synthetic stoppage ball, then degrade_by_game_state.

    Parameters
    ----------
    tracking : Tracking
        Clean tracking of a DFL match.
    profile : GameStateProfile
        The challenge tracking's game-state statistics.
    dead : numpy.ndarray
        (n_frames,) bool, the match's own dead-ball labels.
    seed : int
        Seed passed to both steps.

    Returns
    -------
    Tracking
        The degraded tracking (the dead/alive head's training input).
    """
    return degrade_by_game_state(stoppage_ball(tracking, dead, profile, seed), profile, dead, seed)
