"""The team model: P(H), P(A), P(D) per frame from team shape, trained on the 7 DFL matches.

A histogram gradient-boosted classifier on the SHAPE view of the team features (no ball column), trained
on DFL's raw per-frame labels (H / A / D as delivered) of the speed-bin-degraded DFL tracking. Two training
details:
  - home/away swap augmentation: every training frame is added again with the two teams exchanged and
    the label flipped, so the model learns geometry, not "the home side usually has the ball";
  - class balancing: the total weight of H, A and D is equalised.
`n_members` estimators (seeds seed, seed + 1, ...) are averaged.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import TeamModelConfig
from ..constants import A, H, N_STATES
from ..features.team import shape_columns, swap_teams


class TeamModel:
    """An average of `cfg.n_members` gradient-boosted classifiers of H / A / D on the shape view.

    Parameters
    ----------
    cfg : TeamModelConfig
        Hyperparameters, member count, swap augmentation and class balancing.
    seed : int
        Member i is fitted with ``random_state = seed + i``.

    Attributes
    ----------
    columns_ : list of str
        The shape columns used, set by `fit`.
    estimators_ : list of sklearn.ensemble.HistGradientBoostingClassifier
        The fitted members.
    """
    def __init__(self, cfg: TeamModelConfig, seed: int):
        """Store the configuration; the model is fitted by `fit`."""
        self.cfg = cfg
        self.seed = seed
        self.columns_: list[str] = []
        self.estimators_: list[HistGradientBoostingClassifier] = []

    def _training_set(self, features: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X, y, sample weight) of the labelled frames, swap-augmented and class-balanced per cfg."""
        idx = np.flatnonzero((labels >= 0) & (labels < N_STATES))
        X = features.iloc[idx][self.columns_]
        y = labels[idx].astype(np.int64)
        w = np.ones(len(idx), dtype=np.float64)
        if self.cfg.swap_augment:
            X_swapped = swap_teams(features.iloc[idx])[self.columns_]
            y_swapped = np.where(y == H, A, np.where(y == A, H, y))
            X = np.vstack([X.to_numpy(np.float32), X_swapped.to_numpy(np.float32)])
            y = np.concatenate([y, y_swapped])
            w = np.concatenate([w, w])
        else:
            X = X.to_numpy(np.float32)
        if self.cfg.balance:
            for k in range(N_STATES):
                total = w[y == k].sum()
                if total > 0:
                    w[y == k] *= len(y) / (N_STATES * total)
        return X, y, w

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "TeamModel":
        """Fit the members on the labelled frames.

        Parameters
        ----------
        features : pandas.DataFrame
            One row per training frame (team features); the shape columns (`shape_columns`) are used.
        labels : numpy.ndarray
            (n,) H / A / D codes; any other code (negative, or X) marks an unlabelled frame, which is
            skipped.

        Returns
        -------
        TeamModel
            `self`, fitted.
        """
        self.columns_ = shape_columns(features)
        X, y, w = self._training_set(features, labels)
        self.estimators_ = []
        for member in range(self.cfg.n_members):
            estimator = HistGradientBoostingClassifier(
                max_iter=self.cfg.max_iter, learning_rate=self.cfg.learning_rate,
                max_leaf_nodes=self.cfg.max_leaf_nodes, l2_regularization=self.cfg.l2_regularization,
                early_stopping=False, random_state=self.seed + member)
            self.estimators_.append(estimator.fit(X, y, sample_weight=w))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """(n, 3) P(H), P(A), P(D), the mean over the members.

        Parameters
        ----------
        features : pandas.DataFrame
            One row per frame; must contain `columns_`.

        Returns
        -------
        numpy.ndarray
            (n, 3) float64; a class a member never saw gets 0 from that member.
        """
        X = features[self.columns_].to_numpy(np.float32)
        proba = np.zeros((len(X), N_STATES), dtype=np.float64)
        for estimator in self.estimators_:
            proba[:, estimator.classes_] += estimator.predict_proba(X) / len(self.estimators_)
        return proba
