"""Check that every input file is present and identical to the files the submission was produced from.

    python scripts/check_data.py [--config configs/default.toml] [--quick]

Match files: SHA-256. DFL files: size and MD5 (the checksums figshare publishes). --quick checks sizes only (hashing ~3 GB takes a minute or two). Exit code 1 on any mismatch.
"""

import argparse
import hashlib
import sys
from pathlib import Path

MATCH_FILES = {   # relative to paths.match_data: (bytes, sha256)
    "tactical/BAR-ATH_tactical_atd.txt": (195_714_285, "f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431"),
    "tactical/BAR-ATH_tactical_atd.xml": (59_057, "327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a"),
    "tactical/BAR-ATH_tactical_smart_tagging.xml": (261_055, "0594d35e3df01b9c5002372d8615e2e8ef5379ac8f4e84c5ef7a53f4304c9019"),
}
DFL_FILES = {     # in paths.dfl_data: (bytes, md5)
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml": (11_930, "bc5d5b21dbcf36aa30f4cd0428309f6b"),
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml": (11_722, "37b0f85588c0acfb995fe633c7ac2e0d"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml": (11_745, "b9710781d400dd812f62b0a6e3d7bce8"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml": (11_952, "7b021e577fca917a63111d55cc29ae5b"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml": (11_922, "65a4568e4f374af94d3c1015cbc7c4f5"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml": (11_932, "5979209af316a914ce57347d0633df6c"),
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml": (11_823, "5e2d580d3d8ed97e32959de90d04992b"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml": (418_524_833, "e4273a2adf5dbd9d4ec42def2f01905c"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml": (391_722_803, "bacb2f609c1d1b0bfc906f6b9d4109f2"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml": (348_945_715, "3efff25a86a61768e7d253b4bba137a3"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml": (362_567_859, "4e1063e0a0ae499920cf68ac1aa3ab4e"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml": (371_688_499, "e56ff1336a334f8da51110fbe7cab764"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml": (356_219_423, "9105bad987191d3c3c362c1fce9a4fcd"),
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml": (372_934_140, "e060bb7d7a5a3bfbd15bbf9bc631024d"),
}


def digest(path: Path, algorithm: str) -> str:
    """Hex digest of a file with a ``hashlib`` algorithm ("sha256", "md5"), read in 4 MB chunks."""
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check(folder: Path, expected: dict, algorithm: str, quick: bool) -> list[str]:
    """Check the files of `expected` in `folder`; returns one line per missing or differing file.

    Parameters
    ----------
    folder : Path
        Folder holding the files.
    expected : dict
        File name to ``(bytes, checksum)``, or to a checksum alone (size not checked).
    algorithm : str
        ``hashlib`` algorithm of the checksums.
    quick : bool
        Check sizes only.

    Returns
    -------
    list of str
        Problems found (empty if all match); matching files are printed.
    """
    problems = []
    for name, spec in expected.items():
        size, checksum = spec if isinstance(spec, tuple) else (None, spec)
        path = folder / name
        if not path.is_file():
            problems.append(f"missing   {path}")
        elif size is not None and path.stat().st_size != size:
            problems.append(f"size      {path}: {path.stat().st_size} bytes, expected {size}")
        elif not quick and checksum and digest(path, algorithm) != checksum:
            problems.append(f"checksum  {path}: {algorithm} differs")
        else:
            print(f"ok        {path}")
    return problems


def main() -> None:
    """Command-line entry point: check all inputs, exit code 1 on any problem."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    from possession.config import DEFAULT_CONFIG, load_config
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quick", action="store_true", help="sizes only, no hashing")
    args = parser.parse_args()
    paths = load_config(args.config).paths
    problems = (check(paths.match_data, MATCH_FILES, "sha256", args.quick)
                + check(paths.dfl_data, DFL_FILES, "md5", args.quick))
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("all inputs present and identical to the ones the submission was produced from")


if __name__ == "__main__":
    main()
