"""Fast checks that need no match data: constants, configuration, timeline, decoder, I/O."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import games  # noqa: E402  (the organisers' module at the repository root)
from possession import constants as C, timeline  # noqa: E402
from possession.config import ConfigError, load_config  # noqa: E402
from possession.decoding import viterbi  # noqa: E402
from possession.decoding.playbook import overshoot_dead_edges  # noqa: E402
from possession.io.submission import read_labels, write_submission  # noqa: E402


def test_constants_match_the_organisers_module():
    entry = games.GAMES[games.GAME]
    assert C.GAME == games.GAME
    assert C.N_FRAMES == entry["n_video_frames"]
    assert C.FPS == entry["fps"]
    assert (C.PITCH_LENGTH, C.PITCH_WIDTH) == entry["pitch"]
    assert C.GRADED_HALVES == tuple((a, b + 1) for _, a, b in games.half_video_blocks(games.GAME))


def test_default_config_loads_with_every_key():
    cfg = load_config()
    assert cfg.runtime.seed == 20271117
    assert len(cfg.dfl.matches) == 7
    assert len(cfg.stage1.head.columns) == 40
    assert list(cfg.stage1.cue_rules.weights)[:3] == ["slow", "fast", "ball_out"]


def test_config_rejects_unknown_keys(tmp_path):
    text = (REPO / "configs" / "default.toml").read_text().replace("[runtime]", "[runtime]\nunknown_key = 1")
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="unknown_key"):
        load_config(path)


def test_timeline_context():
    assert timeline.halves() == C.GRADED_HALVES
    with pytest.raises(RuntimeError):
        timeline.kickoffs()
    with timeline.use(timeline.Timeline(halves=((0, 10), (20, 30)), kickoffs=(2, 20))):
        assert timeline.graded_mask(40).sum() == 20
        assert timeline.match_mask(40).sum() == 18
    assert timeline.halves() == C.GRADED_HALVES


TWO_HALVES = timeline.Timeline(halves=((0, 200), (200, 400)))


def _config(**changes) -> viterbi.ViterbiConfig:
    cfg = load_config().viterbi
    return replace(viterbi.ViterbiConfig(**vars(cfg)), **changes)


def test_viterbi_respects_minimum_durations():
    # one long H run with a 5-frame A blip: the blip is shorter than min_duration and must vanish
    proba = np.tile([0.8, 0.1, 0.1], (200, 1))
    proba[100:105] = [0.1, 0.8, 0.1]
    with timeline.use(TWO_HALVES):
        labels = viterbi.decode(np.vstack([proba, proba]), _config(penalty_scale=2.0))
    assert (labels == C.H).all()


def test_viterbi_finds_a_long_change():
    proba = np.tile([0.8, 0.1, 0.1], (200, 1))
    proba[100:] = [0.1, 0.8, 0.1]
    with timeline.use(TWO_HALVES):
        labels = viterbi.decode(np.vstack([proba, proba]), _config(penalty_scale=2.0))[:200]
    assert (labels[:100] == C.H).all() and (labels[100:] == C.A).all()


def test_sandwiched_alive_runs_become_dead():
    block = np.array([C.D] * 10 + [C.H] * 5 + [C.D] * 10, dtype=np.int8)
    assert (viterbi.remove_sandwiched_alive(block, 6) == C.D).all()
    assert (viterbi.remove_sandwiched_alive(block, 5) == block).all()


def test_overshoot_extends_possessions_into_stoppages():
    labels = np.array([C.H] * 50 + [C.D] * 100 + [C.A] * 50, dtype=np.int8)
    with timeline.use(TWO_HALVES):
        out = overshoot_dead_edges(np.r_[labels, labels], frames=25)[:200]
    assert (out[50:75] == C.H).all() and (out[125:150] == C.A).all() and (out[75:125] == C.D).all()


def test_submission_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, C.N_FRAMES).astype(np.int8)
    path = write_submission(codes, tmp_path / "s.csv")
    assert np.array_equal(read_labels(path), codes)
    assert path.read_text().splitlines()[0] == "frame,label"
