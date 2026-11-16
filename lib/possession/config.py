"""Typed configuration, loaded from a TOML file (configs/default.toml is the submitted configuration).

Every pipeline parameter lives in the TOML file; the dataclasses below have no defaults, so a missing,
misspelt or mistyped key is an error (ConfigError), never a silent fallback. The constants that define the dead-ball rules
themselves (physical scales in metres, m/s and seconds) are documented next to each rule in
lib/possession/rules/ and features/deadball.py; they are part of a rule's definition, not knobs.

Relative paths are resolved against the repository root. Two environment variables override paths:
MTAH_DATA_DIR (the organisers' match bundle, as in the official repository) and DFL_DATA_DIR.
"""

import os
import tomllib
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_type_hints

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class PathsConfig:
    """File locations, table ``[paths]``.

    After `load_config` every path is absolute: relative paths are resolved against the repository
    root, and ``MTAH_DATA_DIR`` / ``DFL_DATA_DIR`` override `match_data` / `dfl_data`.

    Attributes
    ----------
    match_data : Path
        Root of the organisers' bundle (contains ``tactical/``).
    dfl_data : Path
        Folder of the DFL open data XML files.
    cache : Path
        Rebuildable intermediate results (safe to delete).
    artifacts : Path
        Trained models, written by training and read by inference.
    submission : Path
        The output CSV.
    """
    match_data: Path          # organisers' bundle root (contains tactical/)
    dfl_data: Path            # the DFL open data XML files
    cache: Path               # rebuildable intermediate results
    artifacts: Path           # trained models
    submission: Path          # the output CSV


@dataclass(frozen=True)
class RuntimeConfig:
    """Reproducibility settings, table ``[runtime]``.

    Attributes
    ----------
    seed : int
        Master seed; every random draw (degradation, gradient boosting) derives from it.
    threads : int
        Number of OpenMP / BLAS threads (>= 1).
    """
    seed: int
    threads: int


@dataclass(frozen=True)
class DflConfig:
    """The DFL training matches, table ``[dfl]``.

    Attributes
    ----------
    matches : tuple of str
        DFL match ids used for training (e.g. ``"J03WMX"``).
    row_stride : int
        Keep one labelled frame in `row_stride` (neighbouring frames are near-copies).
    """
    matches: tuple            # DFL match ids used for training
    row_stride: int           # keep one labelled frame in `row_stride` (neighbours are near-copies)


@dataclass(frozen=True)
class ViterbiDefaults:
    """Defaults of the explicit-duration Viterbi decoder, table ``[viterbi]`` (shared by both stages).

    Attributes
    ----------
    min_duration : tuple of int
        Minimum run length in frames for H, A, D.
    switch : tuple of tuple of float
        (3, 3) relative switch costs ``switch[from][to]``.
    penalty_scale : float
        Unused placeholder: stage 1 calibrates it, stage 2 sets it from ``change_cost_seconds``.
    prior : tuple of float
        (3,) state prior of H, A, D.
    floor : float
        Lower clip of the emission probabilities before taking logs.
    dead_bias : float
        Log-odds (nats) added to the dead state; the start of stage 1's calibration, 0 in stage 2.
    max_sandwiched_alive : int
        Alive runs shorter than this many frames between two D runs become D.
    """
    min_duration: tuple       # frames for H, A, D
    switch: tuple             # relative switch costs [from][to]
    penalty_scale: float      # unused placeholder: stage 1 calibrates it, stage 2 sets it from change_cost_seconds
    prior: tuple
    floor: float
    dead_bias: float
    max_sandwiched_alive: int


@dataclass(frozen=True)
class DeadballHeadConfig:
    """The dead-ball head (gradient boosting on DFL), table ``[stage1.head]``.

    Attributes
    ----------
    max_iter, learning_rate, max_leaf_nodes, l2_regularization
        Hyperparameters of ``HistGradientBoostingClassifier``.
    row_stride : int
        Keep one training frame in `row_stride`.
    lookahead_s : tuple of int
        Look-ahead offsets in seconds; each cue column also gets a copy shifted this far into the future
        (suffix ``__ahead<k>s``).
    columns : tuple of str
        Feature columns the head is trained on, in order.
    """
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    row_stride: int
    lookahead_s: tuple
    columns: tuple


@dataclass(frozen=True)
class CueFeaturesConfig:
    """Ball-out refinements used in the cue features, table ``[stage1.cues]``.

    They switch rules on only in the cues' observed-rule columns (``db_out_*``, ``db_ball_out_seen``, ``db_ball_vanished``); the forced and observed
    layers always use both refinements.

    Attributes
    ----------
    flight_veto : bool
        Apply R2a (an airborne ball projected beyond a line is not out).
    inward_throw : bool
        Apply R4e (a throw first seen mid-air is the restart).
    """
    flight_veto: bool         # R2a in the cues' observed-rule columns (db_out_*)
    inward_throw: bool        # R4e in the cues' observed-rule columns (db_out_*)


@dataclass(frozen=True)
class CueRulesConfig:
    """Hand-set log-odds of the dead-ball cue rules, table ``[stage1.cue_rules]``.

    Attributes
    ----------
    bias : float
        Intercept in nats.
    weights : dict of str to float
        Weight in nats of each cue term (``features.deadball.rule_terms``); the key order is the order
        the terms are summed in.
    """
    bias: float
    weights: dict


@dataclass(frozen=True)
class CalibrationConfig:
    """label-free calibration of the stage-1 decoder, table ``[stage1.calibration]``.

    The decoder's two free parameters are chosen by bisection so that the decoded match meets two
    priors: `dead_bias` for the dead-ball share, `penalty_scale` for the number of live stretches, alternated
    for `rounds` rounds.

    Attributes
    ----------
    target_live_stretches : int
        Target number of live stretches (H / A runs of the neutral-split decode) over the graded match.
    target_dead_share : float
        Target fraction of graded frames decoded as D.
    rounds : int
        Number of (dead share, live stretches) calibration rounds.
    penalty_range : tuple of float
        (low, high) geometric bisection interval of ``penalty_scale``.
    penalty_iterations : int
        Maximum bisection steps for ``penalty_scale``.
    live_stretch_tolerance : int
        Stop when the number of live stretches is this close to the target.
    dead_bias_range : tuple of float
        (low, high) linear bisection interval of ``dead_bias`` in nats.
    dead_bias_iterations : int
        Maximum bisection steps for ``dead_bias``.
    dead_share_tolerance : float
        Stop when the dead share is this close to the target.
    """
    target_live_stretches: int       # label-free priors the stage-1 decoder is calibrated to
    target_dead_share: float
    rounds: int
    penalty_range: tuple          # geometric bisection interval of penalty_scale
    penalty_iterations: int
    live_stretch_tolerance: int   # stop when this close to the live-stretch target
    dead_bias_range: tuple        # linear bisection interval of dead_bias (nats)
    dead_bias_iterations: int
    dead_share_tolerance: float


@dataclass(frozen=True)
class Stage1Config:
    """Stage 1, dead ball vs alive, table ``[stage1]``.

    Attributes
    ----------
    head_weight : float
        Mixing weight in [0, 1]: P(dead) = head_weight * head + (1 - head_weight) * cue rules.
    calibration : CalibrationConfig
        label-free calibration of the decoder.
    overshoot_frames : int
        Frames by which possessions are extended into each neighbouring stoppage (playbook).
    sliver_frames : int
        Alive runs shorter than this many frames left between two D runs after the playbook become D
        (e.g. between observed D and a stoppage).
    keeper_hold_sliver_frames : int
        The same, applied after the goalkeeper-holds-the-ball rule (L5).
    cues : CueFeaturesConfig
        Ball-out refinements used in the cue features.
    head : DeadballHeadConfig
        The DFL-trained dead-ball head.
    cue_rules : CueRulesConfig
        Weights of the hand-set cue rules.
    """
    head_weight: float        # P(dead) = head_weight * head + (1 - head_weight) * cue rules
    calibration: CalibrationConfig
    overshoot_frames: int
    sliver_frames: int
    keeper_hold_sliver_frames: int
    cues: CueFeaturesConfig
    head: DeadballHeadConfig
    cue_rules: CueRulesConfig


@dataclass(frozen=True)
class TeamModelConfig:
    """The stage-2 team model, table ``[team_model]``.

    Attributes
    ----------
    max_iter, learning_rate, max_leaf_nodes, l2_regularization
        Hyperparameters of each ``HistGradientBoostingClassifier`` member.
    n_members : int
        Number of estimators averaged (seeds ``seed``, ``seed + 1``, ...).
    swap_augment : bool
        Add every training frame again with home and away exchanged and the label flipped.
    balance : bool
        Equalise the total sample weight of H, A and D.
    """
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    n_members: int
    swap_augment: bool
    balance: bool


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2, home vs away inside stage 1's live blocks, table ``[stage2]``.

    Attributes
    ----------
    change_cost_seconds : float
        A change of team must be paid for by this many seconds of clear evidence.
    clear_evidence_odds : float
        "Clear" evidence: P(team) / P(other) per frame.
    overshoot_frames : int
        Frames by which possessions are extended into each neighbouring stoppage (playbook).
    sliver_frames : int
        Alive runs shorter than this many frames left between two D runs after the playbook become D
        (e.g. between observed D and a stoppage).
    """
    change_cost_seconds: float    # a change of team must be paid for by this much clear evidence
    clear_evidence_odds: float    # "clear" evidence: P(team) / P(other) per frame
    overshoot_frames: int
    sliver_frames: int


@dataclass(frozen=True)
class Config:
    """The whole pipeline configuration: one attribute per top-level TOML table.

    Attributes
    ----------
    paths : PathsConfig
    runtime : RuntimeConfig
    dfl : DflConfig
    viterbi : ViterbiDefaults
    stage1 : Stage1Config
    team_model : TeamModelConfig
    stage2 : Stage2Config
    """
    paths: PathsConfig
    runtime: RuntimeConfig
    dfl: DflConfig
    viterbi: ViterbiDefaults
    stage1: Stage1Config
    team_model: TeamModelConfig
    stage2: Stage2Config


class ConfigError(ValueError):
    """The configuration file does not match the schema."""


def _freeze(value: Any) -> Any:
    """Turn (nested) lists into (nested) tuples."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _check_scalar(kind: type, value: Any, where: str) -> None:
    """bool must be a TOML boolean, int an integer, float an integer or float, str a string."""
    if kind is bool:
        ok = isinstance(value, bool)
    elif kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind is str:
        ok = isinstance(value, str)
    else:
        ok = True
    if not ok:
        raise ConfigError(f"[{where}] expected {kind.__name__}, got {value!r}")


def _build(cls: type, data: dict, where: str) -> Any:
    """Build dataclass `cls` from a TOML table, checking keys and types recursively."""
    if not isinstance(data, dict):
        raise ConfigError(f"[{where}] expected a table, got {data!r}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown or missing:
        raise ConfigError(f"[{where}] unknown keys {unknown}, missing keys {missing}")
    values = {}
    for name in names:
        kind, raw, here = hints[name], data[name], f"{where}.{name}"
        if is_dataclass(kind):
            values[name] = _build(kind, raw, here)
        elif kind is Path:
            _check_scalar(str, raw, here)
            values[name] = Path(raw)
        elif kind is dict:
            if not isinstance(raw, dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                    for v in raw.values()):
                raise ConfigError(f"[{here}] expected a table of numbers")
            values[name] = dict(raw)          # key order is kept: it is the order the terms are summed in
        elif kind is tuple:
            if not isinstance(raw, list):
                raise ConfigError(f"[{here}] expected an array, got {raw!r}")
            values[name] = _freeze(raw)
        else:
            _check_scalar(kind, raw, here)
            values[name] = raw
    return cls(**values)


def _check_shapes(cfg: "Config") -> None:
    """Cross-field checks the type hints cannot express."""
    v = cfg.viterbi
    if len(v.min_duration) != 3 or len(v.prior) != 3 or len(v.switch) != 3 or any(len(r) != 3 for r in v.switch):
        raise ConfigError("[viterbi] min_duration, prior and switch must be 3 / 3 / 3x3 (H, A, D)")
    c = cfg.stage1.calibration
    if len(c.penalty_range) != 2 or len(c.dead_bias_range) != 2:
        raise ConfigError("[stage1.calibration] penalty_range and dead_bias_range must be [low, high]")
    if not 0 <= cfg.stage1.head_weight <= 1:
        raise ConfigError("[stage1] head_weight must lie in [0, 1]")
    if cfg.runtime.threads < 1:
        raise ConfigError("[runtime] threads must be >= 1")


def _resolve(paths: PathsConfig) -> PathsConfig:
    """Absolute paths, with the MTAH_DATA_DIR / DFL_DATA_DIR overrides applied."""
    def absolute(p: Path) -> Path:
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    match_data = Path(os.environ["MTAH_DATA_DIR"]) if os.environ.get("MTAH_DATA_DIR") else paths.match_data
    dfl_data = Path(os.environ["DFL_DATA_DIR"]) if os.environ.get("DFL_DATA_DIR") else paths.dfl_data
    return PathsConfig(match_data=absolute(match_data), dfl_data=absolute(dfl_data),
                       cache=absolute(paths.cache), artifacts=absolute(paths.artifacts),
                       submission=absolute(paths.submission))


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Read a TOML configuration into a Config (all keys required).

    Parameters
    ----------
    path : Path or str, optional
        The TOML file; defaults to ``configs/default.toml`` (the submitted configuration).

    Returns
    -------
    Config
        The frozen configuration, with every path made absolute (relative paths resolved against the
        repository root; ``MTAH_DATA_DIR`` and ``DFL_DATA_DIR`` override the two data paths).

    Raises
    ------
    ConfigError
        If a key is unknown, missing or of the wrong type, or a cross-field check fails (H / A / D
        shapes of the Viterbi settings, ``[low, high]`` calibration ranges, ``head_weight`` in [0, 1],
        ``threads >= 1``).
    """
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    cfg = _build(Config, data, "config")
    _check_shapes(cfg)
    return replace(cfg, paths=_resolve(cfg.paths))
