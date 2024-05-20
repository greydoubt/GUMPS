# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""This is a wrapper around the StandardScaler and MinMaxScaler from sklearn.preprocessing.
This is done so that dataframes are returned on all transforms instead of
numpy arrays. Eventually that will just be part of sklearn but for now this
is a workaround."""

from abc import ABCMeta, abstractmethod

import pandas as pd
import sklearn.base
import sklearn.preprocessing


class AbstractScaler(metaclass=ABCMeta):
    "sklearn scaler wrapper"
    columns: list[str]
    scaler: "sklearn.base.BaseEstimator|AbstractScaler|None"

    def __init__(self) -> None:
        "abstract scaler init"
        self.columns = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "fit the scaler"

    @property
    def fitted(self) -> bool:
        return len(self.columns) > 0


    def fit_transform(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None, **fit_params) -> pd.DataFrame:
        "fit the scaler and transform the data"
        self.columns = list(X.columns)
        return pd.DataFrame(self.scaler.fit_transform(X, y, **fit_params),
                            columns=self.columns)


    def get_feature_names_out(self, input_features=None):
        "get the feature names"
        return self.scaler.get_feature_names_out(input_features)


    def get_params(self, deep=True):
        "get the parameters"
        return self.scaler.get_params(deep)


    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "transform the data"


    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "inverse transform the data"


    @abstractmethod
    def partial_fit(self, X: pd.DataFrame, y=None):
        "partial fit the data"


    def set_params(self, **params):
        "set the parameters"
        self.scaler.set_params(**params)
        return self
