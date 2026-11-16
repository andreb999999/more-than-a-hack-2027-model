"""Reading and writing `frame,label` CSVs in the format of SUBMISSION.md."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import LABELS, N_FRAMES


def write_submission(codes: np.ndarray, path: Path) -> Path:
    """Write label codes (0..3) as a frame,label CSV with one row per video frame.

    The file has LF line endings on every platform, so it is byte-identical wherever it is produced.

    Parameters
    ----------
    codes : numpy.ndarray
        (N_FRAMES,) integer label codes, 0..3 = H / A / D / X.
    path : Path
        Output CSV; parent folders are created.

    Returns
    -------
    Path
        `path`.

    Raises
    ------
    ValueError
        If `codes` does not have shape (N_FRAMES,) or holds values outside 0..3 or non-integers.
    """
    codes = np.asarray(codes)
    if codes.shape != (N_FRAMES,):
        raise ValueError(f"expected {N_FRAMES} labels, got {codes.shape}")
    if not np.issubdtype(codes.dtype, np.integer) or codes.min() < 0 or codes.max() >= len(LABELS):
        raise ValueError(f"label codes must be integers in 0..{len(LABELS) - 1}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF line endings on every platform, so the file is byte-identical wherever it is produced
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": np.array(LABELS)[codes]}).to_csv(
        path, index=False, lineterminator="\n")
    return path


def read_labels(path: Path) -> np.ndarray:
    """Label codes (0..3) from a frame,label CSV.

    Column names and labels are matched case-insensitively and stripped of whitespace. Rows are taken in
    file order; the ``frame`` column is not checked.

    Parameters
    ----------
    path : Path
        A CSV with a ``label`` column of H / A / D / X.

    Returns
    -------
    numpy.ndarray
        (n_rows,) int8 codes, 0..3 = H / A / D / X.

    Raises
    ------
    ValueError
        If a label is not one of H / A / D / X.
    """
    table = pd.read_csv(path, dtype={"label": "object"})
    table.columns = [str(c).strip().lower() for c in table.columns]
    values = np.array([str(v).strip().upper() for v in table["label"].values])
    codes = np.full(len(values), -1, dtype=np.int8)
    for index, name in enumerate(LABELS):
        codes[values == name] = index
    if (codes < 0).any():
        raise ValueError(f"unknown labels in {path}")
    return codes
