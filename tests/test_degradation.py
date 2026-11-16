"""The speed-bin degradation is deterministic for a seed and leaves its input untouched."""

import numpy as np

from possession.degradation.profiles import SpeedBinProfile
from possession.degradation.simulate import degrade_by_speed_bin
from possession.io.tracking import Tracking


def _clean_tracking(n: int = 600, slots: int = 6, seed: int = 0) -> Tracking:
    rng = np.random.default_rng(seed)
    start = rng.uniform([10, 10], [90, 60], (slots, 2))
    players = start[None] + np.cumsum(rng.normal(0, 0.05, (n, slots, 2)), axis=0)
    ball = np.array([50.0, 34.0]) + np.cumsum(rng.normal(0, 0.2, (n, 2)), axis=0)
    return Tracking(players=players.astype(np.float32), ball=ball.astype(np.float32),
                    team=np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)[:slots],
                    goalkeeper=np.array([True, False, False, True, False, False])[:slots],
                    home_attacks_positive_x_first_half=False)


PROFILE = SpeedBinProfile(missing_share=(0.5, 0.4, 0.3, 0.2), gap_lengths=((50,), (30, 40), (20,), (10,)),
                          slot_run_frames=(100, 150), presence=0.9, ball_sigma=2.0)


def _same(a: Tracking, b: Tracking) -> bool:
    return np.array_equal(a.ball, b.ball, equal_nan=True) and np.array_equal(a.players, b.players, equal_nan=True)


def test_degrade_is_deterministic_for_a_seed():
    clean = _clean_tracking()
    assert _same(degrade_by_speed_bin(clean, PROFILE, seed=3), degrade_by_speed_bin(clean, PROFILE, seed=3))


def test_degrade_differs_for_another_seed():
    clean = _clean_tracking()
    assert not _same(degrade_by_speed_bin(clean, PROFILE, seed=3), degrade_by_speed_bin(clean, PROFILE, seed=4))


def test_degrade_keeps_shapes_and_metadata_and_does_not_touch_the_input():
    clean = _clean_tracking()
    players, ball = clean.players.copy(), clean.ball.copy()
    out = degrade_by_speed_bin(clean, PROFILE, seed=3)
    assert np.array_equal(clean.players, players) and np.array_equal(clean.ball, ball)
    assert out.players.shape == clean.players.shape and out.ball.shape == clean.ball.shape
    assert out.players.dtype == np.float32 and out.ball.dtype == np.float32
    assert np.array_equal(out.team, clean.team) and np.array_equal(out.goalkeeper, clean.goalkeeper)
    assert out.home_attacks_positive_x_first_half is False
    assert np.isnan(out.ball[:, 0]).any() and np.isnan(out.players[:, :, 0]).any()
    assert np.isnan(out.ball[-PROFILE.ball_lead:]).all()
