# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""This is a log combo scaler that is compatible with sklearn.
It is a combination of a log scaler and a StandardScaler or MinMaxScaler.
It implements the same interface as sklearn scalers."""

import pandas as pd

from gumps.scalers.log_scaler import LogScaler
from gumps.scalers.standard_scaler import StandardScaler
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.scaler import AbstractScaler


class LogComboScaler(AbstractScaler):
    "LogComboScaler that is compatible with sklearn"
    scaler: StandardScaler|MinMaxScaler

    def __init__(self, scaler: StandardScaler|MinMaxScaler) -> None:
        "abstract scaler init"
        self.log_scaler = LogScaler()
        self.scaler = scaler

    def get_bounds_transformed(self, lower_bound:pd.Series|None, upper_bound:pd.Series|None) -> tuple[pd.Series, pd.Series]:
        "get the bounds transformed"
        if lower_bound is not None:
            lower_bound = self.log_scaler.transform(lower_bound)
        if upper_bound is not None:
            upper_bound = self.log_scaler.transform(upper_bound)
        return lower_bound, upper_bound

    def fit(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "fit the scaler"
        transformed = self.log_scaler.fit_transform(X, lower_bound=lower_bound, upper_bound=upper_bound)
        lower_bound, upper_bound = self.get_bounds_transformed(lower_bound, upper_bound)
        self.scaler.fit(transformed, lower_bound=lower_bound, upper_bound=upper_bound)
        return self


    def fit_transform(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None) -> pd.DataFrame:
        "fit the scaler and transform the data"
        transformed = self.log_scaler.fit_transform(X, lower_bound=lower_bound, upper_bound=upper_bound)
        lower_bound, upper_bound = self.get_bounds_transformed(lower_bound, upper_bound)
        return self.scaler.fit_transform(transformed, lower_bound=lower_bound, upper_bound=upper_bound)


    def get_feature_names_out(self, input_features=None):
        "get the feature names"
        return self.log_scaler.columns


    def get_params(self, deep=True):
        "get the parameters"
        return {}


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "transform the data"
        transformed = self.log_scaler.transform(X)
        return self.scaler.transform(transformed)


    def inverse_transform(self, X: pd.DataFrame, copy=None) -> pd.DataFrame:
        "inverse transform the data"
        transformed = self.scaler.inverse_transform(X)
        return self.log_scaler.inverse_transform(transformed)


    def partial_fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        "partial fit the data"
        raise NotImplementedError("partial_fit is not implemented")


    def set_params(self, **params):
        "set the parameters"
        return self
