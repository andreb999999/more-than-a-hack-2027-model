"""The DFL open data (Bassek et al. 2025, CC BY 4.0): 7 labelled Bundesliga matches, the training data.

"An integrated dataset of spatiotemporal and event data in elite soccer", figshare 28196177
(doi:10.6084/m9.figshare.28196177). Per match, two files are read (scripts/download_dfl.py):

    DFL_02_01_matchinformation_<comp>_DFL-MAT-<match>.xml        teams, playing positions, pitch size
    DFL_04_03_positions_raw_observed_<comp>_DFL-MAT-<match>.xml  25 Hz positions; every ball frame carries
                                                                 BallPossession (1 home, 2 away) and
                                                                 BallStatus (1 alive, 0 dead)

The labels are therefore the same quantity as the challenge's ground truth (H / A / D per frame),
produced by the data provider. Timeline: the two halves are laid end to end with a GAP of NaN frames
between them (label X), so rolling features never mix halves. Coordinates: DFL is centred; it is
shifted to a corner origin and scaled to the challenge pitch.
"""

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import cache
from ..constants import A, D, H, PITCH_LENGTH, PITCH_WIDTH, X
from .tracking import Tracking

GAP = 750                     # NaN frames between the halves (30 s), labels X
SECTIONS = ("firstHalf", "secondHalf")
GOALKEEPER_POSITION = "TW"    # DFL's PlayingPosition code for the goalkeeper


@dataclass
class DflMatch:
    """One DFL match on its own frame axis (first half, gap, second half).

    Attributes
    ----------
    name : str
        ``"dfl_<match id>"``.
    tracking : Tracking
        Positions in metres on the challenge pitch (corner origin); NaN in the gap between the halves.
    sign : numpy.ndarray
        (n,) float32, +1 where the home team attacks +x, -1 otherwise.
    labels : numpy.ndarray
        (n,) int8 H / A / D / X codes from BallPossession / BallStatus (X in the gap, on frames
        without a ball record, and on alive frames without a possession code).
    halves : tuple
        ``((start, stop), (start, stop))``, half-open frame ranges of the two halves.
    """
    name: str
    tracking: Tracking
    sign: np.ndarray        # (n,) +1 where the home team attacks +x
    labels: np.ndarray      # (n,) H / A / D / X codes from BallPossession / BallStatus
    halves: tuple           # ((start, stop), (start, stop)), half-open

    @property
    def n_frames(self) -> int:
        """Number of frames, gap included."""
        return self.tracking.n_frames


def source_files(dfl_dir: Path, match: str) -> tuple[Path, Path]:
    """(match information, positions) XML files of one DFL match.

    Parameters
    ----------
    dfl_dir : Path
        Folder of the DFL open data.
    match : str
        DFL match id, e.g. ``"J03WMX"``.

    Returns
    -------
    tuple of Path
        The ``DFL_02_01_matchinformation_*`` and ``DFL_04_03_positions_raw_observed_*`` files.

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    """
    return _file(dfl_dir, "02_01_matchinformation", match), _file(dfl_dir, "04_03_positions_raw_observed", match)


def _file(dfl_dir: Path, kind: str, match: str) -> Path:
    """The first DFL file of `kind` for `match` (FileNotFoundError if none)."""
    paths = sorted(Path(dfl_dir).glob(f"DFL_{kind}_*_DFL-MAT-{match}.xml"))
    if not paths:
        raise FileNotFoundError(f"no {kind} file for {match} in {dfl_dir}: run scripts/download_dfl.py")
    return paths[0]


def _match_information(dfl_dir: Path, match: str) -> tuple[dict[str, int], set[str], tuple[float, float]]:
    """Team of every person (0 home, 1 away), the goalkeepers, and the pitch size in metres."""
    root = ET.parse(_file(dfl_dir, "02_01_matchinformation", match)).getroot()
    team_of, keepers = {}, set()
    for team in root.iter("Team"):
        side = 0 if team.get("Role") == "home" else 1
        for player in team.iter("Player"):
            team_of[player.get("PersonId")] = side
            if player.get("PlayingPosition") == GOALKEEPER_POSITION:
                keepers.add(player.get("PersonId"))
    env = root.find(".//Environment")
    return team_of, keepers, (float(env.get("PitchX", 105)), float(env.get("PitchY", 68)))


def _read_positions(dfl_dir: Path, match: str, team_of: dict[str, int]) -> tuple[dict, dict]:
    """Per half: per-person rows (N, x, y) and the ball's rows (N, x, y, possession, status). Streams the
    ~400 MB XML so it never sits in memory whole."""
    people: dict[tuple[str, str], list] = {}
    ball: dict[str, list] = {s: [] for s in SECTIONS}
    current = None
    for event, element in ET.iterparse(_file(dfl_dir, "04_03_positions_raw_observed", match), events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            section, person = element.get("GameSection"), element.get("PersonId")
            if section not in SECTIONS:
                current = None
            elif element.get("TeamId", "").upper() == "BALL":
                current = ("ball", section)
            elif person in team_of:
                current = ("person", section, person)
                people.setdefault((section, person), [])
            else:
                current = None                       # referees
        elif event == "end" and element.tag == "Frame":
            if current is not None:
                n, x, y = int(element.get("N")), float(element.get("X")), float(element.get("Y"))
                if current[0] == "ball":
                    ball[current[1]].append((n, x, y, int(element.get("BallPossession", 0)),
                                             int(element.get("BallStatus", 0))))
                else:
                    people[(current[1], current[2])].append((n, x, y))
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            element.clear()
    return people, ball


def _parse(dfl_dir: Path, match: str, team_of: dict[str, int]) -> dict:
    """Raw arrays of one match: players, ball, labels, period (0 in the gap), persons (slot order)."""
    people, ball = _read_positions(dfl_dir, match, team_of)
    persons = sorted({p for _, p in people})
    slot = {p: i for i, p in enumerate(persons)}
    blocks = []
    for section in SECTIONS:
        frames = np.array([row[0] for row in ball[section]], dtype=np.int64)
        first, n = int(frames.min()), int(frames.max() - frames.min() + 1)
        pl = np.full((n, len(persons), 2), np.nan, dtype=np.float32)
        for (sec, person), rows in people.items():
            if sec == section and rows:
                arr = np.asarray(rows, dtype=np.float64)
                idx = arr[:, 0].astype(np.int64) - first
                keep = (idx >= 0) & (idx < n)
                pl[idx[keep], slot[person]] = arr[keep, 1:3]
        b = np.asarray(ball[section], dtype=np.float64)
        idx = b[:, 0].astype(np.int64) - first
        bxy = np.full((n, 2), np.nan, dtype=np.float32)
        bxy[idx] = b[:, 1:3]
        lab = np.full(n, X, dtype=np.int8)
        alive = b[:, 4] == 1
        lab[idx[~alive]] = D
        lab[idx[alive & (b[:, 3] == 1)]] = H
        lab[idx[alive & (b[:, 3] == 2)]] = A
        blocks.append((pl, bxy, lab))
    gap_pl = np.full((GAP, len(persons), 2), np.nan, dtype=np.float32)
    return {
        "players": np.concatenate([blocks[0][0], gap_pl, blocks[1][0]]),
        "ball": np.concatenate([blocks[0][1], np.full((GAP, 2), np.nan, np.float32), blocks[1][1]]),
        "labels": np.concatenate([blocks[0][2], np.full(GAP, X, np.int8), blocks[1][2]]),
        "period": np.concatenate([np.ones(len(blocks[0][2]), np.int8), np.zeros(GAP, np.int8),
                                  np.full(len(blocks[1][2]), 2, np.int8)]),
        "persons": persons,
    }


def attack_sign(players: np.ndarray, team: np.ndarray, goalkeeper: np.ndarray, period: np.ndarray) -> np.ndarray:
    """+1 where the home team attacks +x: the home keeper stands at the goal it defends.

    Per period, the home goalkeepers' mean x over the period decides: in the half x < L / 2 the home team
    defends the low-x goal, so it attacks +x. Frames outside a period keep +1 (they are never labelled).

    Parameters
    ----------
    players : numpy.ndarray
        (n, n_slots, 2) positions in metres on the challenge pitch.
    team : numpy.ndarray
        (n_slots,) 0 = home, 1 = away.
    goalkeeper : numpy.ndarray
        (n_slots,) bool.
    period : numpy.ndarray
        (n,) int, 1 / 2 for the halves, 0 in the gap.

    Returns
    -------
    numpy.ndarray
        (n,) float32, +1 or -1.

    Raises
    ------
    ValueError
        If there is no home goalkeeper.
    """
    sign = np.ones(len(period), dtype=np.float32)
    gk = np.flatnonzero(goalkeeper & (team == 0))
    if not len(gk):
        raise ValueError("no home goalkeeper: the attack direction cannot be read")
    for p in np.unique(period[period > 0]):
        rows = period == p
        with np.errstate(all="ignore"):
            x = np.nanmean(players[rows][:, gk, 0])
        sign[rows] = 1.0 if x < PITCH_LENGTH / 2 else -1.0
    return sign


def load_match(dfl_dir: Path, match: str, cache_dir: Path) -> DflMatch:
    """One DFL match with tracking in metres on the challenge pitch and labels from BallPossession / BallStatus.

    DFL's centred coordinates are shifted to a corner origin and scaled from the match's pitch size to
    the challenge pitch. The parsed XML is cached (keyed on this module's source and the two files' size and mtime).

    Parameters
    ----------
    dfl_dir : Path
        Folder of the DFL open data.
    match : str
        DFL match id, e.g. ``"J03WMX"``.
    cache_dir : Path
        Cache folder for the parsed arrays.

    Returns
    -------
    DflMatch
        The two halves laid end to end with a gap of ``GAP`` NaN frames (label X) between them.

    Raises
    ------
    FileNotFoundError
        If a source file is missing.
    ValueError
        If the match has no home goalkeeper.
    """
    team_of, keepers, (pitch_x, pitch_y) = _match_information(dfl_dir, match)
    key = cache.fingerprint(sys.modules[__name__], *source_files(dfl_dir, match))
    raw = cache.cached(cache_dir, f"dfl_{match}", key, lambda: _parse(dfl_dir, match, team_of))
    scale = np.array([PITCH_LENGTH / pitch_x, PITCH_WIDTH / pitch_y])
    offset = np.array([pitch_x / 2, pitch_y / 2])
    players = ((raw["players"] + offset) * scale).astype(np.float32)
    ball_xy = ((raw["ball"] + offset) * scale).astype(np.float32)
    persons, period = raw["persons"], raw["period"]
    team = np.array([team_of[p] for p in persons], dtype=np.int8)
    goalkeeper = np.array([p in keepers for p in persons], dtype=bool)
    sign = attack_sign(players, team, goalkeeper, period)
    home_positive = bool(sign[np.flatnonzero(period == 1)[0]] > 0)
    tracking = Tracking(players=players, ball=ball_xy, team=team, goalkeeper=goalkeeper,
                        home_attacks_positive_x_first_half=home_positive)
    first_len = int((period == 1).sum())
    halves = ((0, first_len), (first_len + GAP, len(period)))
    return DflMatch(name=f"dfl_{match}", tracking=tracking, sign=sign, labels=raw["labels"], halves=halves)
