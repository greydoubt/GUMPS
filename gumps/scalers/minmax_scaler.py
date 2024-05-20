# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Wrapper for sklearn MinMaxScaler"

import pandas as pd
import sklearn.preprocessing

from gumps.scalers.scaler import AbstractScaler

class MinMaxScaler(AbstractScaler):
    "wrapper for sklearn MinMaxScaler"
    scaler: sklearn.preprocessing.MinMaxScaler

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False) -> None:
        super().__init__()
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range,
                                                         copy=copy,
                                                         clip=clip)


    def fit(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "fit the scaler"
        self.columns = list(X.columns)

        data = [X,]
        if lower_bound is not None:
            data.append(lower_bound.to_frame().T)
        if upper_bound is not None:
            data.append(upper_bound.to_frame().T)

        data = pd.concat(data, axis=0, ignore_index=True)
        self.scaler.fit(data, y)
        return self


    def partial_fit(self, X: pd.DataFrame, y=None, lower_bound:pd.Series|None=None, upper_bound:pd.Series|None=None):
        "partial fit the data"
        self.columns = list(X.columns)
        self.scaler.partial_fit(X, y)
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "transform the data"
        return pd.DataFrame(self.scaler.transform(X),
                            columns=self.columns)


    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        "inverse transform the data"
        return pd.DataFrame(self.scaler.inverse_transform(X),
                            columns=self.columns)