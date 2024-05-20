# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""This is a null scaler. It does nothing except implement the sklearn scaler interface."""

import pandas as pd

from gumps.scalers.scaler import AbstractScaler

class NullScaler(AbstractScaler):
    "wrapper for sklearn MinMaxScaler"

    def __init__(self) -> None:
        super().__init__()
        self.scaler = None


    def fit(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "fit the scaler"
        self.columns = list(X.columns)
        return self


    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        "fit the scaler and transform the data"
        self.columns = list(X.columns)
        return X.copy()


    def get_feature_names_out(self, input_features=None):
        "get the feature names"
        return self.columns


    def get_params(self, deep=True):
        "get the parameters"
        return {}


    def partial_fit(self, X: pd.DataFrame, y=None):
        "partial fit the data"
        self.columns = list(X.columns)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "transform the data"
        return X.copy()


    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "inverse transform the data"
        return X.copy()


    def set_params(self, **params):
        "set the parameters"
        return self
