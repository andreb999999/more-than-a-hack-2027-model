"""Team model training set (swap augmentation, class balance) and a tiny fit / predict smoke test."""

import numpy as np
import pandas as pd

from possession import constants as C
from possession.config import TeamModelConfig
from possession.features.team import shape_columns
from possession.models.team_model import TeamModel

H, A, D = C.H, C.A, C.D


def _model_config(swap_augment: bool = True, balance: bool = True) -> TeamModelConfig:
    return TeamModelConfig(max_iter=5, learning_rate=0.1, max_leaf_nodes=7, l2_regularization=0.0, n_members=2,
                           swap_augment=swap_augment, balance=balance)


def _features(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "h_cx": rng.uniform(0, 100, n), "a_cx": rng.uniform(0, 100, n),
        "s_vx": rng.normal(size=n), "med_spd": rng.uniform(0, 3, n),
        "ball_vis": rng.integers(0, 2, n).astype(float), "h_d1": rng.uniform(0, 20, n), "a_d1": rng.uniform(0, 20, n),
    }).astype(np.float32)


def test_shape_columns_drop_the_ball_columns():
    assert shape_columns(_features(3)) == ["h_cx", "a_cx", "s_vx", "med_spd"]


def test_training_set_swaps_the_augmented_half():
    features = _features(7)
    labels = np.array([H, H, H, A, D, -1, C.X])
    model = TeamModel(_model_config(balance=False), seed=0)
    model.columns_ = shape_columns(features)
    X, y, w = model._training_set(features, labels)
    kept = features.iloc[:5]
    assert X.shape == (10, 4)
    assert y.tolist() == [H, H, H, A, D, A, A, A, H, D]
    assert np.array_equal(X[:5], kept[model.columns_].to_numpy(np.float32))
    assert np.array_equal(X[5:, 0], kept["a_cx"].to_numpy(np.float32))
    assert np.array_equal(X[5:, 1], kept["h_cx"].to_numpy(np.float32))
    assert np.array_equal(X[5:, 2], -kept["s_vx"].to_numpy(np.float32))
    assert np.array_equal(X[5:, 3], kept["med_spd"].to_numpy(np.float32))
    assert (w == 1).all()


def test_training_set_balances_the_classes():
    features = _features(9)
    labels = np.array([H, H, H, H, H, A, D, D, -1])
    for swap in (True, False):
        model = TeamModel(_model_config(swap_augment=swap), seed=0)
        model.columns_ = shape_columns(features)
        _, y, w = model._training_set(features, labels)
        totals = [w[y == k].sum() for k in range(C.N_STATES)]
        assert np.allclose(totals, len(y) / C.N_STATES)
        assert np.isclose(w.sum(), len(y))
    assert len(y) == 8


def test_training_set_with_a_missing_class():
    features = _features(4)
    labels = np.array([H, H, H, A])
    model = TeamModel(_model_config(swap_augment=False), seed=0)
    model.columns_ = shape_columns(features)
    _, y, w = model._training_set(features, labels)
    assert np.isclose(w[y == H].sum(), w[y == A].sum())
    assert np.isfinite(w).all()


def test_fit_predict_smoke():
    n = 90
    features = _features(n, seed=5)
    labels = np.tile([H, A, D], n // 3)
    model = TeamModel(_model_config(), seed=11).fit(features, labels)
    assert model.columns_ == ["h_cx", "a_cx", "s_vx", "med_spd"]
    assert len(model.estimators_) == 2
    proba = model.predict_proba(features.iloc[:20])
    assert proba.shape == (20, 3)
    assert np.allclose(proba.sum(1), 1.0) and (proba >= 0).all()
    again = TeamModel(_model_config(), seed=11).fit(features, labels).predict_proba(features.iloc[:20])
    assert np.array_equal(proba, again)
