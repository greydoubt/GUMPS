# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper around the scikit-learn KernelPCA solver with automatic parameter scaling"""

import logging
from pathlib import Path
import warnings

import attrs
import joblib
import pandas as pd
import sklearn.decomposition
import sklearn.exceptions
import matplotlib.axes

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.standard_scaler import StandardScaler
from gumps.scalers.null_scaler import NullScaler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class KernelPCASettings:
    """Settings for the KernelPCA solver"""
    input_data: pd.DataFrame
    auto_scaling: bool = True
    n_components: int|None = None
    kernel: str = attrs.field(default="linear")
    @kernel.validator
    def _validate_kernel(self, attribute, value):
        if value not in ["linear", "poly", "rbf", "sigmoid", "cosine"]:
            raise ValueError(f"kernel must be one of linear, poly, rbf, sigmoid, cosine, not {value}")

    gamma: float = 1.0
    degree: int = 3
    coef0: float = 1.0
    alpha: float = 1.0
    fit_inverse_transform: bool = False
    eigen_solver: str = "auto"
    tol: float = 0
    max_iter: int|None = None
    iterated_power: int|str = "auto"
    remove_zero_eig: bool = False
    random_state: int|None = None
    copy_X: bool = True
    n_jobs: int|None = None


class KernelPCA:
    "implement principal component analysis"

    def __init__(self, settings: KernelPCASettings) -> None:
        "Initialize the PCA solver"
        self.settings = settings
        self.pca = self._get_pca()
        self.input_scaler = self._get_scaler()
        self.scaled_input: pd.DataFrame = self._scale_values()
        self.fitted: bool = False


    def _get_pca(self):
        "return the PCA solver"
        return sklearn.decomposition.KernelPCA(n_components=self.settings.n_components,
                                kernel=self.settings.kernel,
                                gamma=self.settings.gamma,
                                degree=self.settings.degree,
                                coef0=self.settings.coef0,
                                alpha=self.settings.alpha,
                                fit_inverse_transform=self.settings.fit_inverse_transform,
                                eigen_solver=self.settings.eigen_solver,
                                tol=self.settings.tol,
                                max_iter=self.settings.max_iter,
                                iterated_power=self.settings.iterated_power,
                                remove_zero_eig=self.settings.remove_zero_eig,
                                random_state=self.settings.random_state,
                                copy_X=self.settings.copy_X,
                                n_jobs=self.settings.n_jobs)


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
        "fit the KernelPCA solver"
        self.pca.fit(self.scaled_input)
        self.fitted = True


    def fit_transform(self):
        "fit the PCA solver and transform the input data"
        self.fit()
        return self.transform(self.settings.input_data)


    def get_params(self):
        "return the parameters"
        return self.pca.get_params()


    def inverse_transform(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "inverse transform the input data"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        scaled_input_data = self.pca.inverse_transform(input_data)
        return self.input_scaler.inverse_transform(scaled_input_data)


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
                raise RuntimeError("Regressor failed to load.") from exc

        return instance


    def get_principal_components(self) -> pd.DataFrame:
        "get the PCA component vectors"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        columns = [f"PC{i}" for i in range(self.pca.eigenvectors_.shape[1])]
        return pd.DataFrame(self.pca.eigenvectors_,
                            columns=columns)


    def get_explained_variance(self) -> pd.DataFrame:
        "get the explained variance ratio for each principal component"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        explained_variance_ratio = self.pca.eigenvalues_ / self.pca.eigenvalues_.sum()

        rows = [f"PC{i}" for i in range(explained_variance_ratio.shape[0])]

        return pd.DataFrame(explained_variance_ratio,
                            columns=["Explained Variance Ratio"],
                            index = rows)


    def get_explained_variance_plot(self) -> matplotlib.axes._axes.Axes:
        "get a plot of the explained variance"
        if not self.fitted:
            raise RuntimeError("PCA solver has not been fitted.")

        return self.get_explained_variance().plot.bar()
