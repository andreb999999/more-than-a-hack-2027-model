"""Run-length helpers on empty input."""

import numpy as np

from possession.features.common import run_bounds, runs, segment_bounds, segments


def test_empty_masks_and_label_sequences_have_no_runs():
    empty = np.zeros(0, dtype=np.int8)
    assert runs(empty.astype(bool)) == []
    assert all(len(a) == 0 for a in run_bounds(empty.astype(bool)))
    assert all(len(a) == 0 for a in segment_bounds(empty))
    assert segments(empty) == []
