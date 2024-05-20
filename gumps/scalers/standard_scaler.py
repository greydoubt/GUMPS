# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Wrapper for sklearn StandardScaler"

import pandas as pd
import sklearn.preprocessing

from gumps.scalers.scaler import AbstractScaler

class StandardScaler(AbstractScaler):
    "wrapper for sklearn StandardScaler"
    scaler: sklearn.preprocessing.StandardScaler

    def __init__(self, *, copy=True, with_mean=True, with_std=True) -> None:
        super().__init__()
        self.scaler = sklearn.preprocessing.StandardScaler(copy=copy,
                                                           with_mean=with_mean,
                                                           with_std=with_std)


    def fit(self, X: pd.DataFrame, y=None, sample_weight=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "fit the scaler"
        self.columns = list(X.columns)
        self.scaler.fit(X, y, sample_weight)
        return self


    def partial_fit(self, X: pd.DataFrame, y=None, sample_weight=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "partial fit the data"
        self.columns = list(X.columns)
        self.scaler.partial_fit(X, y, sample_weight)
        return self


    def transform(self, X: pd.DataFrame, copy=None) -> pd.DataFrame:
        "transform the data"
        return pd.DataFrame(self.scaler.transform(X, copy),
                            columns=self.columns)


    def inverse_transform(self, X: pd.DataFrame, copy=None) -> pd.DataFrame:
        "inverse transform the data"
        return pd.DataFrame(self.scaler.inverse_transform(X, copy),
                            columns=self.columns)
