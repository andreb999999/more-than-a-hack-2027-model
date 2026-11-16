"""Array helpers of features.common and the home/away swap of features.team."""

import numpy as np
import pandas as pd
import pytest

from possession.constants import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W
from possession.features.common import row_quantile, run_bounds, runs, segment_bounds, segments, time_since, time_until
from possession.features.team import swap_teams


def _loop_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _loop_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, int(labels[start])))
            start = i
    return out


@pytest.mark.parametrize("seed", range(8))
def test_runs_match_a_python_loop(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    mask = rng.random(n) < rng.uniform(0.1, 0.9)
    expected = _loop_runs(mask)
    starts, stops = run_bounds(mask)
    assert list(zip(starts.tolist(), stops.tolist())) == expected
    assert [(int(a), int(b)) for a, b in runs(mask)] == expected


def test_runs_edge_cases():
    assert runs(np.zeros(0, dtype=bool)) == []
    assert runs(np.zeros(5, dtype=bool)) == []
    assert [(int(a), int(b)) for a, b in runs(np.ones(5, dtype=bool))] == [(0, 5)]
    assert [(int(a), int(b)) for a, b in runs(np.array([1, 0, 1], dtype=bool))] == [(0, 1), (2, 3)]


@pytest.mark.parametrize("seed", range(8))
def test_segments_match_a_python_loop(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(1, 60))
    labels = rng.integers(0, 4, n).astype(np.int8)
    labels = np.repeat(labels, rng.integers(1, 5, n))[:n]
    expected = _loop_segments(labels)
    starts, stops = segment_bounds(labels)
    assert [(int(a), int(b), int(labels[a])) for a, b in zip(starts, stops)] == expected
    assert [(int(a), int(b), int(c)) for a, b, c in segments(labels)] == expected


def test_segments_of_a_single_value():
    assert [(int(a), int(b), int(c)) for a, b, c in segments(np.array([2, 2, 2]))] == [(0, 3, 2)]
    assert [(int(a), int(b), int(c)) for a, b, c in segments(np.array([1]))] == [(0, 1, 1)]


def test_time_since_counts_seconds_and_caps():
    event = np.zeros(100, dtype=bool)
    event[10] = True
    out = time_since(event, cap_s=2.0)
    assert out.dtype == np.float32
    assert (out[:10] == 2.0).all()
    assert out[10] == 0.0
    assert out[11] == pytest.approx(1 / FPS)
    assert out[10 + 25] == pytest.approx(1.0)
    assert (out[10 + 50:] == 2.0).all()
    assert out.max() == 2.0


def test_time_until_mirrors_time_since():
    event = np.zeros(100, dtype=bool)
    event[[30, 90]] = True
    out = time_until(event, cap_s=1.0)
    assert out[30] == 0.0 and out[90] == 0.0
    assert out[29] == pytest.approx(1 / FPS)
    assert (out[:5] == 1.0).all()
    assert (out[91:] == 1.0).all()
    assert np.allclose(out, time_since(event[::-1], 1.0)[::-1])


def test_time_since_without_any_event_is_the_cap():
    assert (time_since(np.zeros(7, dtype=bool), cap_s=5.0) == 5.0).all()


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
def test_row_quantile_is_the_lower_nan_quantile(q):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(200, 11)).astype(np.float32)
    values[rng.random(values.shape) < 0.3] = np.nan
    values[0] = np.nan
    values[1, 1:] = np.nan
    with pytest.warns(RuntimeWarning):
        expected = np.nanquantile(values, q, axis=1, method="lower").astype(np.float32)
    np.testing.assert_array_equal(row_quantile(values, q), expected)
    assert np.isnan(row_quantile(values, q)[0])


def test_row_quantile_on_a_simple_row():
    values = np.array([[4.0, np.nan, 1.0, 3.0, 2.0]])
    assert row_quantile(values, 0.5)[0] == 2.0
    assert row_quantile(values, 0.9)[0] == 3.0
    assert row_quantile(values, 1.0)[0] == 4.0


def _team_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 20
    return pd.DataFrame({
        "h_cx": rng.uniform(0, L, n), "a_cx": rng.uniform(0, L, n),
        "h_vx__m25": rng.normal(size=n), "a_vx__m25": rng.normal(size=n),
        "s_cx": rng.normal(size=n), "s_vx__d62": rng.normal(size=n),
        "xh_ball": rng.uniform(0, L, n), "yh_ball": rng.uniform(0, W, n),
        "med_spd": rng.uniform(0, 3, n), "ball_vis": rng.integers(0, 2, n).astype(float),
    })


def test_swap_teams_exchanges_and_mirrors():
    frame = _team_frame()
    swapped = swap_teams(frame)
    assert list(swapped.columns) == list(frame.columns)
    assert np.array_equal(swapped["h_cx"], frame["a_cx"]) and np.array_equal(swapped["a_cx"], frame["h_cx"])
    assert np.array_equal(swapped["h_vx__m25"], frame["a_vx__m25"])
    assert np.array_equal(swapped["s_cx"], -frame["s_cx"])
    assert np.array_equal(swapped["s_vx__d62"], -frame["s_vx__d62"])
    assert np.allclose(swapped["xh_ball"], L - frame["xh_ball"])
    assert np.allclose(swapped["yh_ball"], W - frame["yh_ball"])
    assert np.array_equal(swapped["med_spd"], frame["med_spd"])
    assert np.array_equal(swapped["ball_vis"], frame["ball_vis"])


def test_swap_teams_is_an_involution():
    frame = _team_frame()
    frame.index = np.arange(100, 120)
    twice = swap_teams(swap_teams(frame))
    pd.testing.assert_frame_equal(twice, frame, check_exact=False, rtol=1e-12, atol=1e-12)
