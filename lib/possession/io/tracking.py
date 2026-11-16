"""Loading the organisers' match assets into frame-aligned arrays.

Everything is indexed by *video frame* (row i of a submission): the ATD's 1-based `frame_id` i + 1
lands in row i. Only three files of the organisers' bundle are read:

    tactical/BAR-ATH_tactical_atd.xml            FIFA/EPTS metadata (teams, goalkeepers, attack direction)
    tactical/BAR-ATH_tactical_atd.txt            FIFA/EPTS raw tracking, players and ball
    tactical/BAR-ATH_tactical_smart_tagging.xml  Smart Tagging (set pieces, goals, team phases)
"""

import sys
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import cache
from ..constants import FPS, GAME, N_FRAMES, PITCH_LENGTH, PITCH_WIDTH

PLAYER_FIELDS = 3  # x, y, speed per slot in the EPTS text


@dataclass
class Tracking:
    """Frame-aligned tracking of one match.

    Slots are track fragments, not people: a slot carries no identity across time.

    Attributes
    ----------
    players : numpy.ndarray
        (n_frames, n_slots, 2) float32 positions in metres (corner origin), NaN where not observed.
    ball : numpy.ndarray
        (n_frames, 2) float32 ball position in metres, NaN where not observed.
    team : numpy.ndarray
        (n_slots,) int8, 0 = home, 1 = away.
    goalkeeper : numpy.ndarray
        (n_slots,) bool, True for the slots the metadata labels as goalkeepers.
    home_attacks_positive_x_first_half : bool
        Attack direction of the home team in the first half.
    """
    players: np.ndarray
    ball: np.ndarray
    team: np.ndarray
    goalkeeper: np.ndarray
    home_attacks_positive_x_first_half: bool

    @property
    def n_frames(self) -> int:
        """Number of frames (rows of `ball`)."""
        return self.ball.shape[0]


def asset_path(match_data: Path, asset: str) -> Path:
    """Path of one asset of the tactical source in the organisers' bundle (layout of games.py).

    Parameters
    ----------
    match_data : Path
        Root of the organisers' bundle.
    asset : {"atd_meta", "atd_raw", "smart_tagging"}
        Which file: EPTS metadata XML, EPTS raw tracking text, or Smart Tagging XML.

    Returns
    -------
    Path
        ``<match_data>/tactical/BAR-ATH_tactical<suffix>``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    suffix = {"atd_meta": "_atd.xml", "atd_raw": "_atd.txt", "smart_tagging": "_smart_tagging.xml"}[asset]
    path = Path(match_data) / "tactical" / f"{GAME}_tactical{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"match data not found: {path}\nUnpack the organisers' bundle into {match_data} "
                                "(see docs/DATA.md) or set MTAH_DATA_DIR.")
    return path


def _parse_atd_text(raw_path: Path, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised parse of the EPTS text: '<frame_id>:<x,y,s;...>:<bx,by,bs>' per line."""
    text = raw_path.read_text()
    flat = text.replace(":", ",").replace(";", ",").replace("\n", ",").rstrip(",")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        values = np.fromstring(flat, sep=",")
    width = 1 + PLAYER_FIELDS * n_slots + PLAYER_FIELDS
    if values.size % width:
        raise ValueError(f"{raw_path}: {values.size} values is not a multiple of the {width}-wide record")
    table = values.reshape(-1, width)
    frame_id = table[:, 0].astype(np.int64)
    if not np.array_equal(frame_id, np.arange(1, len(frame_id) + 1)):
        raise ValueError(f"{raw_path}: frame_id is not 1..n contiguous")
    players = table[:, 1:1 + PLAYER_FIELDS * n_slots].reshape(-1, n_slots, PLAYER_FIELDS)
    ball = table[:, 1 + PLAYER_FIELDS * n_slots:]
    return players, ball


def _read_metadata(meta_path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """(team per slot, goalkeeper per slot, home attacks +x in the first half) from the EPTS metadata."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"expected two teams in {meta_path}")
    home_id = teams[0].get("id")  # FIFA/EPTS convention: the first team is the home team (DATA.md)
    direction = None
    for parameter in teams[0].iter("ProviderParameter"):
        if parameter.findtext("Name") == "attack_direction_first_half":
            direction = parameter.findtext("Value")
    team, goalkeeper = [], []
    for player in root.iter("Player"):
        team.append(0 if player.get("teamId") == home_id else 1)
        position = None
        for parameter in player.iter("ProviderParameter"):
            if parameter.findtext("Name") == "position_type":
                position = parameter.findtext("Value")
        goalkeeper.append(position == "Goalkeeper")
    if direction not in ("left_to_right", "right_to_left"):
        raise ValueError(f"{meta_path}: no attack_direction_first_half for the home team (found {direction!r})")
    return np.array(team, dtype=np.int8), np.array(goalkeeper, dtype=bool), direction != "right_to_left"


def load_tracking(match_data: Path, cache_dir: Path) -> Tracking:
    """The tactical ATD of the challenge match as a Tracking object (metres, video-frame indexed).

    The ATD is in 0-1 pitch units; positions are scaled by the pitch size. The parsed arrays are
    cached (keyed on this module's source and the two files' size and mtime).

    Parameters
    ----------
    match_data : Path
        Root of the organisers' bundle.
    cache_dir : Path
        Cache folder for the parsed arrays.

    Returns
    -------
    Tracking
        Row i is video frame i (the ATD's ``frame_id`` i + 1).

    Raises
    ------
    FileNotFoundError
        If an ATD file is missing.
    ValueError
        If the metadata or the raw text is malformed, or the match does not have ``N_FRAMES`` frames.
    """
    meta_path, raw_path = asset_path(match_data, "atd_meta"), asset_path(match_data, "atd_raw")
    team, goalkeeper, home_positive = _read_metadata(meta_path)

    def parse() -> dict:
        """Players and ball positions in metres, float32."""
        raw_players, raw_ball = _parse_atd_text(raw_path, len(team))
        scale = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float64)   # the ATD is in 0-1 pitch units
        return {"players": (raw_players[:, :, :2] * scale).astype(np.float32),
                "ball": (raw_ball[:, :2] * scale).astype(np.float32)}

    key = cache.fingerprint(sys.modules[__name__], raw_path, meta_path, PITCH_LENGTH, PITCH_WIDTH)
    arrays = cache.cached(cache_dir, "tracking", key, parse)
    if arrays["players"].shape[0] != N_FRAMES:
        raise ValueError(f"tracking has {arrays['players'].shape[0]} frames, expected {N_FRAMES}")
    return Tracking(players=arrays["players"], ball=arrays["ball"], team=team, goalkeeper=goalkeeper,
                    home_attacks_positive_x_first_half=home_positive)


def load_smart_tagging(match_data: Path) -> pd.DataFrame:
    """Smart Tagging instances: code, start_frame, end_frame (half-open), team, type.

    Parameters
    ----------
    match_data : Path
        Root of the organisers' bundle.

    Returns
    -------
    pandas.DataFrame
        One row per instance. ``code`` is upper-cased (e.g. ``"SET PIECES"``); ``start_frame`` /
        ``end_frame`` are the instance's start / end seconds times FPS, rounded; ``team`` is the
        ``Team`` label (``"N/A"`` when absent) and ``type`` the ``Type`` label (``""`` when absent).

    Raises
    ------
    FileNotFoundError
        If the Smart Tagging file is missing.
    """
    rows = []
    for instance in ET.parse(asset_path(match_data, "smart_tagging")).iter("instance"):
        start_s, end_s = float(instance.findtext("start")), float(instance.findtext("end"))
        tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
        rows.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_frame": int(round(start_s * FPS)),
            "end_frame": int(round(end_s * FPS)),
            "team": (tags.get("Team") or "N/A").strip(),
            "type": (tags.get("Type") or "").strip(),
        })
    return pd.DataFrame(rows)


def empty_tags() -> pd.DataFrame:
    """A Smart Tagging table with no rows (DFL matches have no Smart Tagging).

    Returns
    -------
    pandas.DataFrame
        Empty, with the columns of `load_smart_tagging`.
    """
    return pd.DataFrame(columns=["code", "start_frame", "end_frame", "team", "type"])
