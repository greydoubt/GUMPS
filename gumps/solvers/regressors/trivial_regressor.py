# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of a trivial linear regressor for testing purposes"""

import pandas as pd
import sklearn.linear_model
import optuna

import gumps.solvers.regressors.regression_solver as regression_solver


class TrivialLinearRegressor(regression_solver.AbstractRegressor):
    "This is a trivial linear solver with no save or load functionality and used for writing tests"

    def _get_regressor(self):
        "Return the regressor"
        return sklearn.linear_model.LinearRegression()

    def _fit(self):
        "fit the regressor"
        self.regressor.fit(self.data_regression.scaled_split.train_input,
                           self.data_regression.scaled_split.train_output)
        self.fitted = True
        self.update_error_metrics()

    def _predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        "predict the output data"
        input_data_scaled = self.data_regression.input_scaler.transform(input_data)
        input_data_scaled = pd.DataFrame(input_data_scaled, columns=self.parameters.input_data.columns)

        output_data_scaled = self.regressor.predict(input_data_scaled)
        output_data_scaled = pd.DataFrame(output_data_scaled, columns=self.parameters.output_data.columns)

        output_data = self.data_regression.output_scaler.inverse_transform(output_data_scaled)
        output_data = pd.DataFrame(output_data, columns=self.parameters.output_data.columns)
        return output_data

    def clone(self, parameters: regression_solver.RegressionParameters) -> 'TrivialLinearRegressor':
        "Clone the regressor with new parameters."
        return TrivialLinearRegressor(parameters)

    def auto_tune(self, settings:regression_solver.OptunaParameters) -> None:
        "Automatically tune the regressor."
        raise NotImplementedError("Tuning is not implemented for the trivial regressor")

    def get_tuned_parameters(self) -> dict:
        raise NotImplementedError("Tuning is not implemented for the trivial regressor")
