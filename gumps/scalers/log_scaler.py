# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""This is a log scaler that is compatible with sklearn"""

import pandas as pd
import numpy as np
from gumps.scalers.scaler import AbstractScaler


class LogScaler(AbstractScaler):
    "LogScaler that is compatible with sklearn"

    def __init__(self) -> None:
        "abstract scaler init"
        self.columns: list[str] = []
        self.log_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "fit the scaler"
        self.columns = list(X.columns)

        if lower_bound is None:
            greater_than_zero = (X > 0).all(axis=0)
        else:
            greater_than_zero = lower_bound > 0

        if lower_bound is None:
            lower_bound = X.min()

        if upper_bound is None:
            upper_bound = X.max()

        ratio = upper_bound / lower_bound
        self.log_columns = list(ratio[greater_than_zero & (ratio >= 100)].index)
        return self


    def fit_transform(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None) -> pd.DataFrame:
        "fit the scaler and transform the data"
        self.fit(X, lower_bound=lower_bound, upper_bound=upper_bound)
        return self.transform(X)


    def get_feature_names_out(self, input_features=None):
        "get the feature names"
        return self.columns


    def get_params(self, deep=True):
        "get the parameters"
        return {}


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "transform the data"
        X = X.copy()
        X[self.log_columns] = np.log(X[self.log_columns])
        return X


    def inverse_transform(self, X: pd.DataFrame, copy=None) -> pd.DataFrame:
        "inverse transform the data"
        X = X.copy()
        X[self.log_columns] = np.exp(X[self.log_columns])
        return X


    def partial_fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        "partial fit the data"
        raise NotImplementedError("partial_fit is not implemented")


    def set_params(self, **params):
        "set the parameters"
        return self
