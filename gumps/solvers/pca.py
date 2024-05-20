# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper around the scikit-learn PCA solver with automatic parameter scaling"""

import logging
from pathlib import Path
import warnings

import attrs
import joblib
import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.exceptions
import matplotlib.axes
import seaborn as sns

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.standard_scaler import StandardScaler
from gumps.scalers.null_scaler import NullScaler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class PCASettings:
    """Settings for the PCA solver"""
    input_data: pd.DataFrame
    auto_scaling: bool = True
    n_components: int|float|str|None = None
    copy: bool = True
    whiten: bool = False
    svd_solver: str = "auto"
    tol: float = 0.0
    iterated_power: str|int = "auto"
    n_oversamples: int = 10
    power_iteration_normalizer: str|None = "auto"
    random_state: int|None = None

class PCA:
    "implement principal component analysis"

    def __init__(self, settings: PCASettings) -> None:
        "Initialize the PCA solver"
        self.settings = settings
        self.pca = self._get_pca()
        self.input_scaler = self._get_scaler()
        self.scaled_input: pd.DataFrame = self._scale_values()
        self.fitted: bool = False

    def _get_pca(self):
        "return the PCA solver"
        return sklearn.decomposition.PCA(n_components=self.settings.n_components,
                             copy=self.settings.copy,
                             whiten=self.settings.whiten,
                             svd_solver=self.settings.svd_solver,
                             tol=self.settings.tol,
                             iterated_power=self.settings.iterated_power,
                             n_oversamples=self.settings.n_oversamples,
                             power_iteration_normalizer=self.settings.power_iteration_normalizer,
                             random_state=self.settings.random_state)

    def _get_scaler(self) -> LogComboScaler|NullScaler:
        "return the scaler"
        if self.settings.auto_scaling:
            return LogComboScaler(StandardScaler())
        else:
            return NullScaler()

    def _scale_values(self) -> pd.DataFrame:
        "scale the input data and fit the scaler"
        scaled_input_data = self.input_scaler.fit_transform(self.settings.input_data)
        return scaled_input_data

    def fit(self):
        "fit the PCA solver"
        self.pca.fit(self.scaled_input)
        self.fitted = True

    def fit_transform(self):
        "fit the PCA solver and transform the input data"
        self.fit()
        return self.transform(self.settings.input_data)

    def get_covariance(self):
        "return the covariance matrix"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        return pd.DataFrame(self.pca.get_covariance(),
                            columns=self.settings.input_data.columns,
                            index=self.settings.input_data.columns)

    def get_params(self):
        "return the parameters"
        return self.pca.get_params()

    def get_precision(self):
        "return the precision matrix"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        return pd.DataFrame(self.pca.get_precision(),
                            columns=self.settings.input_data.columns,
                            index=self.settings.input_data.columns)

    def inverse_transform(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "inverse transform the input data"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        scaled_input_data = self.pca.inverse_transform(input_data)
        return self.input_scaler.inverse_transform(scaled_input_data)

    def score(self, input_data: pd.DataFrame) -> float:
        "score the input data"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        scaled_input_data = self.input_scaler.transform(input_data)
        return self.pca.score(scaled_input_data)

    def score_samples(self, input_data: pd.DataFrame) -> np.ndarray:
        "score the input data"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        scaled_input_data = self.input_scaler.transform(input_data)
        return self.pca.score_samples(scaled_input_data)

    def set_params(self, **kwargs):
        "set the parameters"
        self.pca.set_params(**kwargs)

    def transform(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "transform the input data"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        scaled_input_data = self.input_scaler.transform(input_data)
        transformed_data = self.pca.transform(scaled_input_data)

        columns = [f"PC{i}" for i in range(transformed_data.shape[1])]
        return pd.DataFrame(transformed_data,
                            columns=columns)

    def save(self, path_dir: Path) -> None:
        "save the regressor, input scaler and output scaler"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted yet")

        joblib.dump(self.pca, path_dir / "pca.joblib")
        joblib.dump(self.input_scaler, path_dir / "input_scaler.joblib")
        joblib.dump(self.settings, path_dir / "settings.joblib")
        joblib.dump(self.scaled_input, path_dir / "scaled_input.joblib")


    @classmethod
    def load_instance(cls, path_dir:Path):
        "load the pca solver, input scaler and output scaler"
        instance = cls.__new__(cls)
        instance.fitted = True

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=sklearn.exceptions.InconsistentVersionWarning)
            warnings.simplefilter("error", UserWarning)
            try:
                instance.pca = joblib.load(path_dir / "pca.joblib")
            except (UserWarning, sklearn.exceptions.InconsistentVersionWarning) as exc:
                raise RuntimeError("Failed to load pca") from exc

            try:
                instance.input_scaler = joblib.load(path_dir / "input_scaler.joblib")
            except (UserWarning, sklearn.exceptions.InconsistentVersionWarning) as exc:
                raise RuntimeError("Failed to load input scaler") from exc

            instance.settings = joblib.load(path_dir / "settings.joblib")
            instance.scaled_input = joblib.load(path_dir / "scaled_input.joblib")

        return instance


    @classmethod
    def rebuild_model(cls, path_dir: Path, auto_resave: bool = True):
        "Rebuild the regressor and optionally resave it"
        logger.warning("PCA failed to load, rebuilding PCA.")
        settings = joblib.load(path_dir / "settings.joblib")
        instance = cls(settings)
        instance.fit()

        if auto_resave:
            logger.warning("PCA was rebuilt, resaving PCA.")
            instance.save(path_dir)

        return instance


    @classmethod
    def load(cls, path_dir: Path, auto_rebuild: bool = True, auto_resave: bool = True):
        "load the regressor, input scaler and output scaler"
        try:
            instance = cls.load_instance(path_dir)
        except RuntimeError as exc:
            if auto_rebuild:
                instance = cls.rebuild_model(path_dir, auto_resave=auto_resave)
            else:
                raise RuntimeError("PCA failed to load.") from exc

        return instance


    def get_principal_components(self) -> pd.DataFrame:
        "get the PCA component vectors"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        columns = [f"PC{i}" for i in range(self.pca.components_.shape[0])]
        return pd.DataFrame(self.pca.components_.T,
                            columns=columns,
                            index=self.settings.input_data.columns)


    def get_explained_variance(self) -> pd.DataFrame:
        "get the explained variance ratio for each principal component"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        rows = [f"PC{i}" for i in range(self.pca.explained_variance_ratio_.shape[0])]

        return pd.DataFrame(self.pca.explained_variance_ratio_,
                            columns=["Explained Variance Ratio"],
                            index = rows)


    def get_explained_variance_plot(self) -> matplotlib.axes._axes.Axes:
        "get a plot of the explained variance"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        return self.get_explained_variance().plot.bar()


    def get_covariance_plot(self) -> matplotlib.axes._axes.Axes:
        "get a plot of the covariance"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        heat=sns.heatmap(self.get_covariance(), cmap='coolwarm')
        heat.set_yticklabels(heat.get_yticklabels(), rotation=30, fontsize=16)
        heat.set_xticklabels(heat.get_xticklabels(), rotation=30, fontsize=16)

        return heat
