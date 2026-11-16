"""Viterbi decoder, layered decoding and the playbook on synthetic probabilities."""

from dataclasses import replace

import numpy as np
import pytest

from possession import constants as C, timeline
from possession.config import load_config
from possession.decoding import viterbi
from possession.decoding.layered import compose, decode_with_layers
from possession.decoding.playbook import apply_playbook, free_frames, overshoot_dead_edges

H, A, D, X = C.H, C.A, C.D, C.X


def _config(**changes) -> viterbi.ViterbiConfig:
    return replace(viterbi.ViterbiConfig(**vars(load_config().viterbi)), **changes)


def _proba(labels: np.ndarray, strength: float = 0.9) -> np.ndarray:
    weak = (1 - strength) / 2
    proba = np.full((len(labels), 3), weak)
    proba[np.arange(len(labels)), np.minimum(labels, D)] = strength
    return proba


def test_possession_count_counts_alive_segments_per_half():
    labels = np.array([H, H, D, D, A, A, H, H, D, D] + [X] * 10 + [D, H, H, H, H, H, H, A, A, D], dtype=np.int8)
    with timeline.use(timeline.Timeline(halves=((0, 10), (20, 30)))):
        assert viterbi.possession_count(labels) == 3 + 2
    with timeline.use(timeline.Timeline(halves=((0, 5), (5, 10)))):
        assert viterbi.possession_count(np.full(10, H, dtype=np.int8)) == 2
    with timeline.use(timeline.Timeline(halves=((0, 10), (10, 10)))):
        assert viterbi.possession_count(np.full(10, D, dtype=np.int8)) == 0


def test_dead_share_excludes_pre_kickoff_frames():
    labels = np.full(200, H, dtype=np.int8)
    labels[0:20] = D
    labels[100:120] = D
    labels[50:60] = D
    with timeline.use(timeline.Timeline(halves=((0, 100), (100, 200)), kickoffs=(20, 120))):
        assert viterbi.dead_share(labels) == pytest.approx(10 / 160)


def test_frames_outside_the_halves_are_x():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=150)
    with timeline.use(timeline.Timeline(halves=((10, 60), (80, 130)))):
        labels = viterbi.decode(proba, _config(min_duration=(3, 3, 5), penalty_scale=1.0))
    outside = np.r_[0:10, 60:80, 130:150]
    assert (labels[outside] == X).all()
    assert (labels[10:60] != X).all() and (labels[80:130] != X).all()
    assert labels.dtype == np.int8


def test_empty_second_half_does_not_crash():
    proba = _proba(np.r_[np.full(100, H), np.full(100, A)])
    with timeline.use(timeline.Timeline(halves=((0, 200), (200, 200)))):
        labels = viterbi.decode(proba, _config(min_duration=(5, 5, 10), penalty_scale=2.0))
        count = viterbi.possession_count(labels)
    assert (labels[:100] == H).all() and (labels[100:] == A).all()
    assert count == 2


def _noisy_possessions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = np.repeat(rng.integers(0, 3, n), rng.integers(10, 80, n))[:n]
    proba = np.zeros((n, 3))
    proba[np.arange(n), truth] = 0.3
    proba += rng.uniform(0, 1, (n, 3))
    return proba / proba.sum(1, keepdims=True)


def test_calibrate_penalty_scale_reaches_the_target():
    proba = _noisy_possessions(1600, seed=1)
    config = _config(min_duration=(5, 5, 10), max_sandwiched_alive=0)
    with timeline.use(timeline.Timeline(halves=((0, 800), (800, 1600)))):
        many = viterbi.possession_count(viterbi.decode(proba, replace(config, penalty_scale=0.01)))
        few = viterbi.possession_count(viterbi.decode(proba, replace(config, penalty_scale=50.0)))
        assert many > few + 20
        target = (many + few) // 2
        calibrated = viterbi.calibrate_penalty_scale(proba, config, target, 0.01, 50.0, iters=30, tol=1)
        count = viterbi.possession_count(viterbi.decode(proba, calibrated))
    assert 0.01 <= calibrated.penalty_scale <= 50.0
    assert abs(count - target) <= 2
    assert calibrated.min_duration == config.min_duration


def test_calibrate_dead_bias_moves_the_dead_share():
    proba = _noisy_possessions(1600, seed=2)
    config = _config(min_duration=(5, 5, 10), max_sandwiched_alive=0, penalty_scale=2.0)
    with timeline.use(timeline.Timeline(halves=((0, 800), (800, 1600)), kickoffs=(0, 800))):
        calibrated = viterbi.calibrate_dead_bias(proba, config, 0.5, -8.0, 8.0, iters=30, tol=0.01)
        share = viterbi.dead_share(viterbi.decode(proba, calibrated))
    assert abs(share - 0.5) <= 0.03


LAYER_TIMELINE = timeline.Timeline(halves=((0, 300), (300, 600)), kickoffs=(0, 300))


def _layers(hard: slice | None = None, dead: slice | None = None, alive: slice | None = None
            ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    def mask(where: slice | None) -> np.ndarray:
        out = np.zeros(600, dtype=bool)
        if where is not None:
            out[where] = True
        return out

    return (None if hard is None else mask(hard)), mask(dead), mask(alive)


def test_hard_dead_frames_are_dead():
    proba = _proba(np.full(600, H))
    hard, dead, alive = _layers(hard=slice(100, 150))
    config = _config(min_duration=(12, 12, 25), penalty_scale=2.0)
    with timeline.use(LAYER_TIMELINE):
        labels, used = decode_with_layers(proba, config, hard, dead, alive, overshoot_frames=0, sliver_frames=0)
    assert (labels[100:150] == D).all()
    assert (labels[:100] == H).all() and (labels[150:] == H).all()
    assert used == config


def test_hard_dead_is_trimmed_by_the_overshoot_but_certain_dead_is_not():
    proba = _proba(np.full(600, H))
    config = _config(min_duration=(12, 12, 25), penalty_scale=2.0)
    with timeline.use(LAYER_TIMELINE):
        hard, dead, alive = _layers(hard=slice(100, 200))
        trimmed, _ = decode_with_layers(proba, config, hard, dead, alive, overshoot_frames=25, sliver_frames=0)
        hard, dead, alive = _layers(dead=slice(100, 200))
        kept, _ = decode_with_layers(proba, config, hard, dead, alive, overshoot_frames=25, sliver_frames=0)
    assert (trimmed[100:125] == H).all() and (trimmed[175:200] == H).all() and (trimmed[125:175] == D).all()
    assert (kept[100:200] == D).all()
    assert (kept[:100] == H).all() and (kept[200:] == H).all()


def test_certain_alive_frames_are_never_dead_unless_hard_dead():
    labels = np.full(600, H)
    labels[150:300] = D
    labels[450:600] = D
    proba = _proba(labels, 0.95)
    hard, dead, alive = _layers(hard=slice(500, 530), alive=slice(150, 300))
    alive[480:560] = True
    config = _config(min_duration=(12, 12, 25), penalty_scale=2.0, max_sandwiched_alive=0)
    with timeline.use(LAYER_TIMELINE):
        out, _ = decode_with_layers(proba, config, hard, dead, alive, overshoot_frames=0, sliver_frames=0)
        free, _ = decode_with_layers(proba, config, None, dead, np.zeros(600, dtype=bool),
                                     overshoot_frames=0, sliver_frames=0)
    assert (free[150:300] == D).all()
    assert (out[150:300] != D).all()
    assert (out[500:530] == D).all()
    assert (out[(alive & ~hard)] != D).all()


def test_decode_with_layers_keeps_the_team_split_on_certain_alive_frames():
    labels = np.r_[np.full(300, H), np.full(300, A)]
    proba = _proba(labels, 0.6)
    proba[:, D] = 0.6
    proba /= proba.sum(1, keepdims=True)
    alive = np.ones(600, dtype=bool)
    config = _config(min_duration=(12, 12, 25), penalty_scale=2.0)
    with timeline.use(LAYER_TIMELINE):
        out, _ = decode_with_layers(proba, config, None, np.zeros(600, dtype=bool), alive,
                                    overshoot_frames=0, sliver_frames=0)
    assert (out[:300] == H).all() and (out[300:] == A).all()


def test_compose_rows_sum_to_one():
    rng = np.random.default_rng(4)
    proba = compose(rng.uniform(-0.2, 1.2, 50), rng.uniform(-0.2, 1.2, 50))
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(1), 1.0) and (proba >= 0).all()


def test_free_frames_dead_outside_play_and_kickoff_to_the_following_team():
    labels = np.full(300, A, dtype=np.int8)
    labels[20:30] = D
    labels[160:170] = D
    labels[170:250] = H
    with timeline.use(timeline.Timeline(halves=((10, 100), (150, 250)), kickoffs=(20, 160))):
        out = free_frames(labels)
    assert (out[:20] == D).all()
    assert (out[100:160] == D).all()
    assert (out[250:] == D).all()
    assert (out[20:30] == A).all() and (out[30:100] == A).all()
    assert (out[160:170] == H).all() and (out[170:250] == H).all()
    assert (labels[20:30] == D).all()


def test_free_frames_leaves_a_live_kickoff_frame_alone():
    labels = np.full(100, H, dtype=np.int8)
    labels[30:40] = A
    with timeline.use(timeline.Timeline(halves=((0, 50), (50, 100)), kickoffs=(30, 50))):
        out = free_frames(labels)
    assert (out[:30] == D).all() and (out[30:40] == A).all() and (out[40:] == H).all()


def test_overshoot_keeps_min_dead():
    labels = np.array([H] * 50 + [D] * 40 + [A] * 50, dtype=np.int8)
    with timeline.use(timeline.Timeline(halves=((0, 140), (140, 140)))):
        out = overshoot_dead_edges(labels, frames=25, min_dead=25)
    assert (out[:57] == H).all() and (out[57:83] == D).all() and (out[83:] == A).all()
    short = np.array([H] * 50 + [D] * 20 + [A] * 50, dtype=np.int8)
    with timeline.use(timeline.Timeline(halves=((0, 120), (120, 120)))):
        assert np.array_equal(overshoot_dead_edges(short, frames=25, min_dead=25), short)


def test_overshoot_one_sided_runs():
    leading = np.array([D] * 100 + [H] * 100, dtype=np.int8)
    trailing = np.array([A] * 100 + [D] * 100, dtype=np.int8)
    with timeline.use(timeline.Timeline(halves=((0, 200), (200, 400)))):
        out = overshoot_dead_edges(np.r_[leading, trailing], frames=25)
    assert (out[:75] == D).all() and (out[75:200] == H).all()
    assert (out[200:325] == A).all() and (out[325:400] == D).all()


def test_overshoot_ignores_x_neighbours_and_stays_inside_halves():
    labels = np.array([H] * 20 + [X] * 10 + [D] * 60 + [H] * 10, dtype=np.int8)
    with timeline.use(timeline.Timeline(halves=((30, 90), (90, 100)))):
        out = overshoot_dead_edges(labels, frames=25)
    assert np.array_equal(out, labels)


def test_apply_playbook_without_overshoot_is_free_frames():
    labels = np.r_[np.full(50, D), np.full(100, H), np.full(50, D), np.full(100, A)].astype(np.int8)
    with timeline.use(timeline.Timeline(halves=((0, 150), (150, 300)), kickoffs=(10, 160))):
        assert np.array_equal(apply_playbook(labels, 0), free_frames(labels))
        out = apply_playbook(labels, 25)
    assert (out[:10] == D).all() and (out[10:150] == H).all()
    assert (out[150:160] == D).all() and (out[160:] == A).all()
