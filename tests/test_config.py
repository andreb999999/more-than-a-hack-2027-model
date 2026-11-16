"""Configuration schema: missing, unknown and mistyped keys, array freezing, path resolution."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from possession.config import DEFAULT_CONFIG, REPO_ROOT, ConfigError, load_config
from possession.features.deadball import cue_rule_probability, rule_terms

DEFAULT_TEXT = DEFAULT_CONFIG.read_text()


def _variant(tmp_path: Path, old: str, new: str) -> Path:
    assert DEFAULT_TEXT.count(old) == 1, old
    path = tmp_path / "variant.toml"
    path.write_bytes(DEFAULT_TEXT.replace(old, new).encode())
    return path


def test_missing_key_is_an_error(tmp_path):
    path = _variant(tmp_path, "seed = 20271117", "")
    with pytest.raises(ConfigError, match="missing keys \\['seed'\\]"):
        load_config(path)


def test_nested_unknown_key_is_an_error(tmp_path):
    path = _variant(tmp_path, "[stage1.calibration]\n", "[stage1.calibration]\nbogus_knob = 1\n")
    with pytest.raises(ConfigError, match="stage1.calibration.*bogus_knob"):
        load_config(path)


def test_unknown_table_is_an_error(tmp_path):
    path = _variant(tmp_path, "[runtime]\n", "[extra]\nx = 1\n\n[runtime]\n")
    with pytest.raises(ConfigError, match="extra"):
        load_config(path)


@pytest.mark.parametrize(("old", "new"), [
    ("threads = 4 ", "threads = 4.5 "),
    ("threads = 4 ", "threads = true "),
    ("seed = 20271117", "seed = \"20271117\""),
    ("flight_veto = false", "flight_veto = \"false\""),
    ("flight_veto = false", "flight_veto = 0"),
    ("head_weight = 0.75", "head_weight = \"0.75\""),
    ("cache = \"cache\"", "cache = 1"),
    ("row_stride = 15                         #", "row_stride = [15]                       #"),
    ("bias = -1.5", "bias = [-1.5]"),
    ("slow = 3.0", "slow = \"3.0\""),
    ("slow = 3.0", "slow = true"),
])
def test_wrong_types_are_errors(tmp_path, old, new):
    with pytest.raises(ConfigError):
        load_config(_variant(tmp_path, old, new))


def test_an_integer_is_accepted_for_a_float(tmp_path):
    cfg = load_config(_variant(tmp_path, "head_weight = 0.75", "head_weight = 1"))
    assert cfg.stage1.head_weight == 1


@pytest.mark.parametrize("value", ["[0.01]", "[0.01, 1.0, 400.0]"])
def test_calibration_ranges_need_two_elements(tmp_path, value):
    with pytest.raises(ConfigError, match="penalty_range"):
        load_config(_variant(tmp_path, "penalty_range = [0.01, 400.0]", f"penalty_range = {value}"))


def test_dead_bias_range_needs_two_elements(tmp_path):
    with pytest.raises(ConfigError, match="dead_bias_range"):
        load_config(_variant(tmp_path, "dead_bias_range = [-8.0, 8.0]", "dead_bias_range = [-8.0]"))


def test_cross_field_checks(tmp_path):
    with pytest.raises(ConfigError, match="head_weight"):
        load_config(_variant(tmp_path, "head_weight = 0.75", "head_weight = 1.5"))
    with pytest.raises(ConfigError, match="threads"):
        load_config(_variant(tmp_path, "threads = 4 ", "threads = 0 "))
    with pytest.raises(ConfigError, match="min_duration"):
        load_config(_variant(tmp_path, "min_duration = [12, 12, 100]", "min_duration = [12, 100]"))


def test_arrays_become_tuples():
    cfg = load_config()
    assert isinstance(cfg.dfl.matches, tuple)
    assert isinstance(cfg.viterbi.switch, tuple) and all(isinstance(row, tuple) for row in cfg.viterbi.switch)
    assert cfg.viterbi.min_duration == (12, 12, 100)
    assert cfg.stage1.calibration.penalty_range == (0.01, 400.0)
    assert isinstance(cfg.stage1.head.columns, tuple)
    assert isinstance(cfg.stage1.cue_rules.weights, dict)


def test_relative_paths_become_absolute_under_the_repo_root(monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    monkeypatch.delenv("DFL_DATA_DIR", raising=False)
    paths = load_config().paths
    for value in vars(paths).values():
        assert isinstance(value, Path) and value.is_absolute()
    assert paths.match_data == (REPO_ROOT / "data").resolve()
    assert paths.dfl_data == (REPO_ROOT / "data" / "external" / "dfl").resolve()
    assert paths.cache == (REPO_ROOT / "cache").resolve()
    assert paths.submission == (REPO_ROOT / "outputs" / "submission.csv").resolve()


def test_environment_overrides_the_data_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path / "bundle"))
    monkeypatch.setenv("DFL_DATA_DIR", str(tmp_path / "dfl"))
    paths = load_config().paths
    assert paths.match_data == tmp_path / "bundle"
    assert paths.dfl_data == tmp_path / "dfl"
    assert paths.cache == (REPO_ROOT / "cache").resolve()


def test_relative_environment_paths_resolve_under_the_repo_root(monkeypatch):
    monkeypatch.setenv("MTAH_DATA_DIR", "elsewhere")
    monkeypatch.setenv("DFL_DATA_DIR", "")
    paths = load_config().paths
    assert paths.match_data == (REPO_ROOT / "elsewhere").resolve()
    assert paths.dfl_data == (REPO_ROOT / "data" / "external" / "dfl").resolve()


RULE_TERM_INPUTS = (
    "db_spd_med_3s", "db_spd_p90_3s", "db_spd_p90_1s", "db_out_until_restart", "db_ball_vis_1s", "db_ball_still_2s",
    "db_wall_n", "db_wall_ring", "db_since_restart", "db_ball_spd_1s", "db_out_evidence", "db_kickoff", "db_corner",
    "db_goal_kick", "db_throw_in", "db_sp_open", "db_goal_pending", "db_lost_slow", "db_keeper_holds",
    "db_player_carries", "db_walk",
)


def test_cue_rule_weights_name_exactly_the_rule_terms():
    cues = pd.DataFrame(np.zeros((4, len(RULE_TERM_INPUTS)), dtype=np.float32), columns=list(RULE_TERM_INPUTS))
    terms = rule_terms(cues)
    weights = load_config().stage1.cue_rules.weights
    assert list(weights) == list(terms.columns)
    p = cue_rule_probability(terms, weights, load_config().stage1.cue_rules.bias)
    assert p.shape == (4,) and np.all((p > 0) & (p < 1))
