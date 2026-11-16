"""Submission writer, EPTS text and metadata parsers, and the DFL loader on tiny synthetic files."""

import numpy as np
import pytest

from possession import constants as C
from possession.io import dfl
from possession.io.submission import read_labels, write_submission
from possession.io.tracking import _parse_atd_text, _read_metadata, empty_tags

H, A, D, X = C.H, C.A, C.D, C.X


@pytest.mark.parametrize("codes", [
    np.zeros(10, dtype=np.int8),
    np.zeros((C.N_FRAMES, 1), dtype=np.int8),
    np.zeros(C.N_FRAMES + 1, dtype=np.int8),
])
def test_write_submission_rejects_a_wrong_shape(tmp_path, codes):
    with pytest.raises(ValueError, match="expected"):
        write_submission(codes, tmp_path / "s.csv")
    assert not (tmp_path / "s.csv").exists()


@pytest.mark.parametrize("bad", [4, -1, 0.5])
def test_write_submission_rejects_bad_codes(tmp_path, bad):
    codes = np.zeros(C.N_FRAMES, dtype=np.float64 if isinstance(bad, float) else np.int64)
    codes[100] = bad
    with pytest.raises(ValueError, match="label codes"):
        write_submission(codes, tmp_path / "s.csv")


def test_write_submission_rejects_float_codes_even_when_integral(tmp_path):
    with pytest.raises(ValueError, match="label codes"):
        write_submission(np.zeros(C.N_FRAMES), tmp_path / "s.csv")


def test_write_submission_writes_lf_only(tmp_path):
    codes = np.tile(np.arange(4, dtype=np.int8), C.N_FRAMES // 4 + 1)[:C.N_FRAMES]
    path = write_submission(codes, tmp_path / "nested" / "s.csv")
    data = path.read_bytes()
    assert b"\r\n" not in data
    assert data.startswith(b"frame,label\n0,H\n1,A\n2,D\n3,X\n")
    assert data.count(b"\n") == C.N_FRAMES + 1
    assert np.array_equal(read_labels(path), codes)


def test_read_labels_rejects_unknown_labels(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(b"frame,label\n0,H\n1,Q\n")
    with pytest.raises(ValueError, match="unknown labels"):
        read_labels(path)


def test_read_labels_tolerates_case_and_spaces(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes(b"Frame , Label\n0, h\n1,a \n2,D\n3,x\n")
    assert read_labels(path).tolist() == [H, A, D, X]


ATD_TEXT = (
    "1:0.1,0.2,1.5;0.3,0.4,2.5:0.5,0.6,0.0\n"
    "2:0.11,0.21,1.6;nan,nan,nan:0.51,0.61,3.0\n"
    "3:0.12,0.22,1.7;0.32,0.42,2.7:nan,nan,nan\n"
)


def test_parse_atd_text(tmp_path):
    path = tmp_path / "atd.txt"
    path.write_bytes(ATD_TEXT.encode())
    players, ball = _parse_atd_text(path, n_slots=2)
    assert players.shape == (3, 2, 3) and ball.shape == (3, 3)
    assert np.allclose(players[0], [[0.1, 0.2, 1.5], [0.3, 0.4, 2.5]])
    assert np.allclose(players[2, 0], [0.12, 0.22, 1.7])
    assert np.isnan(players[1, 1]).all() and not np.isnan(players[1, 0]).any()
    assert np.allclose(ball[:2], [[0.5, 0.6, 0.0], [0.51, 0.61, 3.0]])
    assert np.isnan(ball[2]).all()


def test_parse_atd_text_without_a_final_newline(tmp_path):
    path = tmp_path / "atd.txt"
    path.write_bytes(ATD_TEXT.rstrip("\n").encode())
    players, _ = _parse_atd_text(path, n_slots=2)
    assert players.shape == (3, 2, 3)


def test_parse_atd_text_rejects_a_wrong_slot_count(tmp_path):
    path = tmp_path / "atd.txt"
    path.write_bytes(ATD_TEXT.encode())
    with pytest.raises(ValueError, match="multiple"):
        _parse_atd_text(path, n_slots=3)


def test_parse_atd_text_rejects_non_contiguous_frames(tmp_path):
    path = tmp_path / "atd.txt"
    path.write_bytes(ATD_TEXT.replace("3:", "4:", 1).encode())
    with pytest.raises(ValueError, match="contiguous"):
        _parse_atd_text(path, n_slots=2)


def _metadata(direction: str | None = "right_to_left", teams: int = 2) -> str:
    direction_xml = "" if direction is None else (
        "<ProviderParameter><Name>attack_direction_first_half</Name>"
        f"<Value>{direction}</Value></ProviderParameter>")
    team_xml = "".join(
        f'<Team id="T{i}"><Name>Team {i}</Name><ProviderParameters>'
        + (direction_xml if i == 1 else "")
        + "<ProviderParameter><Name>other</Name><Value>x</Value></ProviderParameter>"
        + "</ProviderParameters></Team>"
        for i in range(1, teams + 1))

    def player(pid: str, team: str, position: str) -> str:
        return (f'<Player id="{pid}" teamId="{team}"><Name>{pid}</Name><ProviderParameters>'
                f"<ProviderParameter><Name>position_type</Name><Value>{position}</Value></ProviderParameter>"
                "</ProviderParameters></Player>")

    players = (player("p1", "T2", "Goalkeeper") + player("p2", "T1", "Defender")
               + player("p3", "T1", "Goalkeeper") + player("p4", "T2", "Forward"))
    return (f'<?xml version="1.0" encoding="UTF-8"?><Metadata><Teams>{team_xml}</Teams>'
            f"<Players>{players}</Players></Metadata>")


def test_read_metadata(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_bytes(_metadata("right_to_left").encode())
    team, goalkeeper, home_positive = _read_metadata(path)
    assert team.tolist() == [1, 0, 0, 1] and team.dtype == np.int8
    assert goalkeeper.tolist() == [True, False, True, False]
    assert home_positive is False
    path.write_bytes(_metadata("left_to_right").encode())
    assert _read_metadata(path)[2] is True


def test_read_metadata_requires_the_attack_direction(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_bytes(_metadata(None).encode())
    with pytest.raises(ValueError, match="attack_direction_first_half"):
        _read_metadata(path)


def test_read_metadata_requires_two_teams(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_bytes(_metadata(teams=3).encode())
    with pytest.raises(ValueError, match="two teams"):
        _read_metadata(path)


def test_empty_tags_has_the_columns():
    assert list(empty_tags().columns) == ["code", "start_frame", "end_frame", "team", "type"]
    assert len(empty_tags()) == 0


MATCH = "TEST01"

DFL_INFO = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest>
  <MatchInformation>
    <General MatchId="DFL-MAT-TEST01"/>
    <Environment PitchX="105.0" PitchY="68.0"/>
    <Teams>
      <Team TeamId="DFL-CLU-H" Role="home">
        <Players>
          <Player PersonId="P1" PlayingPosition="TW"/>
          <Player PersonId="P2" PlayingPosition="ZD"/>
        </Players>
      </Team>
      <Team TeamId="DFL-CLU-A" Role="guest">
        <Players>
          <Player PersonId="P3" PlayingPosition="TW"/>
          <Player PersonId="P4" PlayingPosition="STZ"/>
        </Players>
      </Team>
    </Teams>
  </MatchInformation>
</PutDataRequest>
"""


def _frameset(section: str, team: str, person: str, frames: list[str]) -> str:
    body = "\n".join(f"      {f}" for f in frames)
    return f'    <FrameSet GameSection="{section}" TeamId="{team}" PersonId="{person}">\n{body}\n    </FrameSet>\n'


def _player_frames(start: int, n: int, x: float, y: float) -> list[str]:
    return [f'<Frame N="{start + i}" X="{x}" Y="{y}"/>' for i in range(n)]


def _positions() -> str:
    ball_1 = [
        '<Frame N="10000" X="0.0" Y="0.0" BallPossession="1" BallStatus="1"/>',
        '<Frame N="10001" X="1.0" Y="0.0" BallPossession="2" BallStatus="1"/>',
        '<Frame N="10003" X="2.0" Y="1.0" BallPossession="2" BallStatus="0"/>',
        '<Frame N="10004" X="-52.5" Y="-34.0" BallPossession="1" BallStatus="1"/>',
    ]
    ball_2 = [f'<Frame N="{100000 + i}" X="0.0" Y="0.0" BallPossession="{1 + i % 2}" BallStatus="{int(i != 2)}"/>'
              for i in range(6)]
    sets = [
        _frameset("firstHalf", "BALL", "DFL-OBJ-BALL", ball_1),
        _frameset("firstHalf", "DFL-CLU-H", "P1", _player_frames(10000, 5, -50.0, 0.0)),
        _frameset("firstHalf", "DFL-CLU-H", "P2", _player_frames(9998, 8, 0.0, 0.0)),
        _frameset("firstHalf", "DFL-CLU-A", "P3", _player_frames(10000, 5, 50.0, 0.0)),
        _frameset("firstHalf", "DFL-CLU-A", "P4", _player_frames(10001, 3, 10.0, -5.0)),
        _frameset("firstHalf", "referee", "R1", _player_frames(10000, 5, 5.0, 5.0)),
        _frameset("secondHalf", "BALL", "DFL-OBJ-BALL", ball_2),
        _frameset("secondHalf", "DFL-CLU-H", "P1", _player_frames(100000, 6, 50.0, 0.0)),
        _frameset("secondHalf", "DFL-CLU-H", "P2", _player_frames(100000, 6, -10.0, 0.0)),
        _frameset("secondHalf", "DFL-CLU-A", "P3", _player_frames(100000, 6, -50.0, 0.0)),
        _frameset("secondHalf", "DFL-CLU-A", "P4", _player_frames(100000, 6, 10.0, 0.0)),
    ]
    return '<?xml version="1.0" encoding="utf-8"?>\n<PutDataRequest>\n  <Positions>\n' + "".join(sets) + "  </Positions>\n</PutDataRequest>\n"


def _write_dfl(folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_bytes(DFL_INFO.encode())
    (folder / f"DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_bytes(_positions().encode())


def test_dfl_source_files_are_found_or_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dfl"):
        dfl.source_files(tmp_path, MATCH)
    _write_dfl(tmp_path / "dfl")
    info, positions = dfl.source_files(tmp_path / "dfl", MATCH)
    assert info.name.startswith("DFL_02_01_matchinformation") and positions.name.startswith("DFL_04_03")


def test_dfl_load_match(tmp_path):
    _write_dfl(tmp_path / "dfl")
    match = dfl.load_match(tmp_path / "dfl", MATCH, tmp_path / "cache")
    first, second = 5, 6
    n = first + dfl.GAP + second
    assert match.name == f"dfl_{MATCH}"
    assert match.n_frames == n
    assert match.halves == ((0, first), (first + dfl.GAP, n))
    assert match.labels.tolist()[:first] == [H, A, X, D, H]
    assert (match.labels[first:first + dfl.GAP] == X).all()
    assert match.labels.tolist()[first + dfl.GAP:] == [H, A, D, A, H, A]
    tracking = match.tracking
    assert tracking.players.shape == (n, 4, 2) and tracking.ball.shape == (n, 2)
    assert tracking.team.tolist() == [0, 0, 1, 1]
    assert tracking.goalkeeper.tolist() == [True, False, True, False]
    assert tracking.home_attacks_positive_x_first_half is True
    assert (match.sign[:first] == 1).all() and (match.sign[first + dfl.GAP:] == -1).all()
    assert (match.sign[first:first + dfl.GAP] == 1).all()
    assert np.allclose(tracking.ball[0], [C.PITCH_LENGTH / 2, C.PITCH_WIDTH / 2])
    assert np.allclose(tracking.ball[4], [0.0, 0.0], atol=1e-4)
    assert np.isnan(tracking.ball[2]).all()
    assert np.isnan(tracking.ball[first:first + dfl.GAP]).all()
    assert np.isnan(tracking.players[first:first + dfl.GAP]).all()
    assert np.allclose(tracking.players[:first, 1], [C.PITCH_LENGTH / 2, C.PITCH_WIDTH / 2])
    assert np.isnan(tracking.players[0, 3]).all() and np.isnan(tracking.players[4, 3]).all()
    assert np.allclose(tracking.players[1, 3], [(10.0 + 52.5) * C.PITCH_LENGTH / 105, (-5.0 + 34.0) * C.PITCH_WIDTH / 68])
    assert list((tmp_path / "cache").glob(f"dfl_{MATCH}_*.pkl"))
    again = dfl.load_match(tmp_path / "dfl", MATCH, tmp_path / "cache")
    assert np.array_equal(again.labels, match.labels)
    assert np.array_equal(again.tracking.players, tracking.players, equal_nan=True)


def test_dfl_attack_sign_needs_a_home_goalkeeper():
    players = np.zeros((4, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="goalkeeper"):
        dfl.attack_sign(players, np.array([0, 1]), np.array([False, True]), np.array([1, 1, 2, 2]))
