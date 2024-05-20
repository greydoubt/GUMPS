# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#Implement mutual_info_regression from scikit_learn to calculate mutual information between the input and output data
#This will give us the correlation between variables without assuming a linear relationship

import attrs
import pandas as pd
import numpy as np
import sklearn.feature_selection
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from gumps.utilities.smoothing import LPointProblem

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.standard_scaler import StandardScaler


@attrs.define
class MutualSettings:
    input_data: pd.DataFrame
    output_data: pd.DataFrame
    n_neighbors: int|str = attrs.field(default="auto")
    @n_neighbors.validator
    def _check_n_neighbors(self, _, value):
        if isinstance(value, int) and value < 1:
            raise ValueError(f"n_neighbors must be greater than 0, not {value}")
        elif isinstance(value, str) and value != 'auto':
            raise ValueError(f"n_neighbors must be 'auto', not {value}")

    random_state: int | None = None

class MutualInfoRegression:
    "Implement mutual information regression"

    def __init__(self, settings: MutualSettings) -> None:
        "Initialize the mutual information regression solver"
        self.settings = settings
        self.input_scaler, self.output_scaler = self._get_scaler()
        self.scaled_input, self.scaled_output = self._scale_values()
        self.fitted: bool = False
        self.mutual_info_scaled: pd.DataFrame | None = None
        self.update_elbow_point()


    def _get_scaler(self) -> tuple[LogComboScaler, LogComboScaler]:
        "return the scaler"
        return (LogComboScaler(StandardScaler()), LogComboScaler(StandardScaler()))


    def _scale_values(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        "scale the input data and fit the scaler"
        scaled_input_data = self.input_scaler.fit_transform(self.settings.input_data)
        scaled_output_data = self.output_scaler.fit_transform(self.settings.output_data)
        return scaled_input_data, scaled_output_data


    def fit(self) -> None:
        "fit the mutual information regression solver"
        self.mutual_info_scaled = self._fit(self.scaled_input, self.scaled_output)
        self.fitted = True


    def _fit(self, input_data: pd.DataFrame, output_data: pd.DataFrame) -> pd.DataFrame:
        "fit the mutual information regression solver"
        columns = {}
        for column in output_data.columns:
            mutual_info = sklearn.feature_selection.mutual_info_regression(input_data,
                                                                          output_data[column],
                                                                          n_neighbors=self.settings.n_neighbors,
                                                                          random_state=self.settings.random_state)
            mutual_info = pd.Series(mutual_info, index=input_data.columns)
            columns[column] = mutual_info
        return pd.DataFrame(columns)


    def score(self) -> pd.DataFrame:
        "return the mutual information"
        if not self.fitted:
            raise RuntimeError("The mutual information regression solver must be fitted before scoring.")
        return self.mutual_info_scaled


    def score_scaled(self) -> pd.DataFrame:
        "return the scaled mutual information"
        if not self.fitted:
            raise RuntimeError("The mutual information regression solver must be fitted before scoring.")
        return self.mutual_info_scaled


    def plot(self):
        "plot the mutual information"
        if not self.fitted:
            raise RuntimeError("The mutual information regression solver must be fitted before plotting.")

        data = self.score()
        data_scaled = data/data.max()

        heat=sns.heatmap(data_scaled, cmap='coolwarm')
        heat.set_yticklabels(heat.get_yticklabels(), rotation=30, fontsize=16)
        heat.set_xticklabels(heat.get_xticklabels(), rotation=30, fontsize=16)

        return heat


    def plot_scaled(self):
        "plot the scaled mutual information"
        if not self.fitted:
            raise RuntimeError("The mutual information regression solver must be fitted before plotting.")

        data = self.score_scaled()
        data_scaled = data/data.max()

        heat=sns.heatmap(data_scaled, cmap='coolwarm')
        heat.set_yticklabels(heat.get_yticklabels(), rotation=30, fontsize=16)
        heat.set_xticklabels(heat.get_xticklabels(), rotation=30, fontsize=16)

        return heat

    def get_elbow_sample(self) -> pd.DataFrame:
        errors = []
        ks = list(range(1, 30))

        for k in ks:
            reg = KNeighborsRegressor(n_neighbors=k)
            reg.fit(self.scaled_input, self.scaled_output)
            errors.append(np.mean((self.scaled_output - reg.predict(self.scaled_input))**2))

        return pd.DataFrame({'k': ks, 'error': errors})

    def get_elbow_point(self) -> int:
        df = self.get_elbow_sample()

        problem = LPointProblem(df['k'].to_numpy(), -df['error'].to_numpy())
        return int(problem.LPoint()[0])

    def update_elbow_point(self):
        if self.settings.n_neighbors == 'auto':
            self.settings.n_neighbors = self.get_elbow_point()