# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#create an errror metrics class that will be used to cache and evaluate the performance of a regression model
import sklearn.metrics
import sklearn.preprocessing
import attrs
import pandas as pd
from pathlib import Path
import joblib

@attrs.define
class ErrorScalar:
    r_squared: float
    mean_squared_error: float
    root_mean_squared_error: float
    normalized_mean_squared_error: float
    normalized_root_mean_squared_error: float

    def series_output(self) -> pd.Series:
        return pd.Series({"score": self.r_squared,
                          'r2_score': self.r_squared,
                          'mean_squared_error': self.mean_squared_error,
                          'mse': self.mean_squared_error,
                          'root_mean_squared_error': self.root_mean_squared_error,
                          'rmse': self.root_mean_squared_error,
                          'normalized_mean_squared_error': self.normalized_mean_squared_error,
                          'nmse': self.normalized_mean_squared_error,
                          'normalized_root_mean_squared_error': self.normalized_root_mean_squared_error,
                          'nrmse': self.normalized_root_mean_squared_error})

@attrs.define
class ErrorVector:
    r_squared: pd.Series
    mean_squared_error: pd.Series
    root_mean_squared_error: pd.Series
    normalized_mean_squared_error: pd.Series
    normalized_root_mean_squared_error: pd.Series

    def series_output(self) -> pd.Series:
        return pd.Series({"score": self.r_squared.to_numpy().mean(),
                          'r2_score': self.r_squared.to_numpy().mean(),
                          'mean_squared_error': self.mean_squared_error.to_numpy(),
                          'mse': self.mean_squared_error.to_numpy(),
                          'root_mean_squared_error': self.root_mean_squared_error.to_numpy(),
                          'rmse': self.root_mean_squared_error.to_numpy(),
                          'normalized_mean_squared_error': self.normalized_mean_squared_error.to_numpy(),
                          'nmse': self.normalized_mean_squared_error.to_numpy(),
                          'normalized_root_mean_squared_error': self.normalized_root_mean_squared_error.to_numpy(),
                          'nrmse': self.normalized_root_mean_squared_error.to_numpy()})

    def frame_output(self) -> pd.DataFrame:
        return pd.DataFrame({"score": self.r_squared,
                             'r2_score': self.r_squared,
                             'mean_squared_error': self.mean_squared_error,
                             'mse': self.mean_squared_error,
                             'root_mean_squared_error': self.root_mean_squared_error,
                             'rmse': self.root_mean_squared_error,
                             'normalized_mean_squared_error': self.normalized_mean_squared_error,
                             'nmse': self.normalized_mean_squared_error,
                             'normalized_root_mean_squared_error': self.normalized_root_mean_squared_error,
                             'nrmse': self.normalized_root_mean_squared_error}).T


class ErrorMetrics():
    error_scalar: ErrorScalar|None
    error_vector: ErrorVector|None

    def __init__(self, y_true:pd.DataFrame|None = None, y_pred:pd.DataFrame|None = None):
        if y_true is not None and y_pred is not None:
            self.set_data(y_true, y_pred)
        else:
            self.error_scalar = None
            self.error_vector = None

    def set_data(self, y_true:pd.DataFrame, y_pred:pd.DataFrame):
        self.error_scalar = self.get_scalar_error(y_true, y_pred)
        self.error_vector = self.get_vector_error(y_true, y_pred)

    def get_scalar_error(self, y_true:pd.DataFrame, y_pred:pd.DataFrame) -> ErrorScalar:
        r_squared = float(sklearn.metrics.r2_score(y_true, y_pred, multioutput="uniform_average"))
        mean_squared_error = float(sklearn.metrics.mean_squared_error(y_true, y_pred, multioutput="uniform_average"))
        root_mean_squared_error = mean_squared_error**0.5

        #apply normalization to the error metrics by scaling the output_data from 0 to 1
        scaler = sklearn.preprocessing.MinMaxScaler()
        y_true_scaled = scaler.fit_transform(y_true)
        y_pred_scaled = scaler.transform(y_pred)

        normalized_mean_squared_error = float(sklearn.metrics.mean_squared_error(y_true_scaled, y_pred_scaled, multioutput="uniform_average"))
        normalized_root_mean_squared_error = normalized_mean_squared_error**0.5
        return ErrorScalar(r_squared, mean_squared_error, root_mean_squared_error, normalized_mean_squared_error, normalized_root_mean_squared_error)

    def get_vector_error(self, y_true:pd.DataFrame, y_pred:pd.DataFrame) -> ErrorVector:
        r_squared = pd.Series(sklearn.metrics.r2_score(y_true,
                                                       y_pred,
                                                       multioutput="raw_values"),
                                                       index=y_true.columns)

        mean_squared_error = pd.Series(sklearn.metrics.mean_squared_error(y_true,
                                                                          y_pred,
                                                                          multioutput="raw_values"),
                                                                          index=y_true.columns)
        root_mean_squared_error = mean_squared_error**0.5

        #apply normalization to the error metrics by scaling the output_data from 0 to 1
        scaler = sklearn.preprocessing.MinMaxScaler()
        y_true_scaled = scaler.fit_transform(y_true)
        y_pred_scaled = scaler.transform(y_pred)

        normalized_mean_squared_error = pd.Series(sklearn.metrics.mean_squared_error(y_true_scaled,
                                                                                     y_pred_scaled,
                                                                                     multioutput="raw_values"),
                                                                                     index=y_true.columns)
        normalized_root_mean_squared_error = normalized_mean_squared_error**0.5
        return ErrorVector(r_squared, mean_squared_error, root_mean_squared_error, normalized_mean_squared_error, normalized_root_mean_squared_error)


    def save(self, path_dir:Path) -> None:
        if self.error_scalar is None or self.error_vector is None:
            raise ValueError("ErrorMetrics has not been set with data")

        joblib.dump(self.error_scalar, path_dir / "error_scalar.joblib")
        joblib.dump(self.error_vector, path_dir / "error_vector.joblib")


    def load(self, path_dir:Path) -> None:
        self.error_scalar = joblib.load(path_dir / "error_scalar.joblib")
        self.error_vector = joblib.load(path_dir / "error_vector.joblib")


    def has_metrics(self) -> bool:
        return self.error_scalar is not None and self.error_vector is not None


    def get_metrics(self, multioutput:str) -> pd.Series:
        if multioutput not in {"raw_values", "uniform_average"}:
            raise ValueError("multioutput must be 'raw_values' or 'uniform_average'")

        if multioutput == "raw_values" and self.error_vector is not None:
            return self.error_vector.series_output()
        elif multioutput == "uniform_average" and self.error_scalar is not None:
            return self.error_scalar.series_output()
        else:
            raise ValueError("ErrorMetrics has not been set with data")


    def get_metrics_frame(self) -> pd.DataFrame:
        if self.error_vector is not None:
            return self.error_vector.frame_output()
        else:
            raise ValueError("ErrorMetrics has not been set with data")
