"""Stage 2 helpers: change penalty, structured probabilities, P(home | alive)."""

from dataclasses import replace

import numpy as np
import pytest

from possession import constants as C
from possession.config import load_config
from possession.pipeline import stage2

H, A, D = C.H, C.A, C.D


def test_change_penalty_of_the_default_config():
    assert stage2.change_penalty(load_config().stage2) == pytest.approx(1 * 25 * np.log(9) / 2)


def test_change_penalty_scales_with_its_inputs():
    cfg = load_config().stage2
    assert stage2.change_penalty(replace(cfg, change_cost_seconds=2.0)) == pytest.approx(25 * np.log(9))
    assert stage2.change_penalty(replace(cfg, clear_evidence_odds=1.0)) == pytest.approx(0.0)


def test_structured_proba_rows_sum_to_one_with_d_on_dead_frames():
    rng = np.random.default_rng(0)
    p_home = rng.uniform(0, 1, 300)
    dead = np.zeros(300, dtype=bool)
    dead[100:120] = True
    out = stage2.structured_proba(p_home, dead)
    assert out.shape == (300, 3)
    assert np.allclose(out.sum(1), 1.0)
    assert np.allclose(out[dead, D], 0.998) and np.allclose(out[~dead, D], 0.001)


def test_structured_proba_only_clips_p_home():
    rng = np.random.default_rng(1)
    p_home = rng.uniform(-0.5, 1.5, 50)
    out = stage2.structured_proba(p_home, np.zeros(50, dtype=bool))
    assert np.allclose(out[:, H] / (out[:, H] + out[:, A]), np.clip(p_home, 0.001, 0.999))


def test_p_home_given_alive_handles_zeros():
    proba = np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 0.6], [0.0, 0.5, 0.5], [0.2, 0.0, 0.8]])
    assert np.allclose(stage2.p_home_given_alive(proba), [0.0, 0.75, 0.0, 1.0])
    assert np.isfinite(stage2.p_home_given_alive(np.zeros((3, 3)))).all()
