"""The active match timeline and the on-disk cache."""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from possession import cache, constants as C, timeline


def test_nested_use_restores_the_outer_timeline():
    outer = timeline.Timeline(halves=((0, 10), (20, 30)), kickoffs=(2, 21))
    inner = timeline.Timeline(halves=((0, 5), (5, 9)), kickoffs=(0, 5))
    with timeline.use(outer):
        with timeline.use(inner):
            assert timeline.halves() == ((0, 5), (5, 9))
            assert timeline.kickoffs() == (0, 5)
        assert timeline.halves() == ((0, 10), (20, 30))
        assert timeline.kickoffs() == (2, 21)
    assert timeline.halves() == C.GRADED_HALVES


def test_use_restores_the_timeline_after_an_exception():
    outer = timeline.Timeline(halves=((0, 10), (20, 30)), kickoffs=(2, 21))
    with timeline.use(outer):
        with pytest.raises(ZeroDivisionError):
            with timeline.use(timeline.Timeline(halves=((0, 1), (1, 2)))):
                1 / 0
        assert timeline.active().halves == ((0, 10), (20, 30))
    assert timeline.halves() == C.GRADED_HALVES
    with pytest.raises(RuntimeError):
        timeline.kickoffs()


def test_use_normalises_numpy_integers():
    with timeline.use(timeline.Timeline(halves=((np.int64(0), np.int64(4)), (6, 9)), kickoffs=(np.int32(1), 6))):
        assert timeline.halves() == ((0, 4), (6, 9))
        assert all(type(k) is int for k in timeline.kickoffs())


def test_graded_and_match_masks():
    with timeline.use(timeline.Timeline(halves=((2, 6), (8, 12)), kickoffs=(3, 8))):
        graded = timeline.graded_mask(14)
        match = timeline.match_mask(14)
    assert np.flatnonzero(graded).tolist() == [2, 3, 4, 5, 8, 9, 10, 11]
    assert np.flatnonzero(match).tolist() == [3, 4, 5, 8, 9, 10, 11]
    assert not (match & ~graded).any()


def _load_module(path: Path, source: str) -> ModuleType:
    path.write_bytes(source.encode())
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fingerprint_changes_with_a_value():
    assert cache.fingerprint(1, "a") == cache.fingerprint(1, "a")
    assert cache.fingerprint(1, "a") != cache.fingerprint(2, "a")
    assert cache.fingerprint(1.0) != cache.fingerprint(1)
    assert len(cache.fingerprint("x")) == 14


def test_fingerprint_changes_with_a_file_size_and_mtime(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    first = cache.fingerprint(path)
    assert cache.fingerprint(path) == first
    path.write_bytes(b"abcd")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    resized = cache.fingerprint(path)
    assert resized != first
    os.utime(path, (1_700_000_100, 1_700_000_100))
    assert cache.fingerprint(path) != resized


def test_fingerprint_ignores_file_contents_of_equal_size_and_mtime(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    first = cache.fingerprint(path)
    path.write_bytes(b"xyz")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    assert cache.fingerprint(path) == first


def test_fingerprint_changes_with_a_module_source(tmp_path):
    one = _load_module(tmp_path / "fp_module_one.py", "VALUE = 1\n")
    two = _load_module(tmp_path / "fp_module_two.py", "VALUE = 2\n")
    same = _load_module(tmp_path / "fp_module_three.py", "VALUE = 1\n")
    assert cache.fingerprint(one) != cache.fingerprint(two)
    assert cache.fingerprint(one) == cache.fingerprint(same)
    assert cache.fingerprint(timeline) != cache.fingerprint(cache)


def test_cached_builds_once_then_loads(tmp_path):
    calls = []

    def build() -> dict:
        calls.append(1)
        return {"array": np.arange(5), "name": "x"}

    folder = tmp_path / "cache" / "nested"
    first = cache.cached(folder, "table", "k1", build)
    second = cache.cached(folder, "table", "k1", build)
    assert len(calls) == 1
    assert np.array_equal(first["array"], second["array"]) and second["name"] == "x"
    assert (folder / "table_k1.pkl").is_file()
    assert not list(folder.glob("*.tmp"))
    cache.cached(folder, "table", "k2", build)
    assert len(calls) == 2


def test_cached_returns_the_stored_value_not_a_rebuild(tmp_path):
    cache.cached(tmp_path, "value", "key", lambda: 1)
    assert cache.cached(tmp_path, "value", "key", lambda: 2) == 1
