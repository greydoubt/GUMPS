# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""sklearn based Gaussian Process Regressor."""

import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import (ConstantKernel, Kernel, Matern,
                                              WhiteKernel)
import pandas as pd
import attrs
import optuna

import gumps.solvers.regressors.regression_solver as regression_solver
from gumps.scalers.standard_scaler import StandardScaler
from gumps.scalers.log_combo_scaler import LogComboScaler


@attrs.define(kw_only=True)
class GaussianRegressionParameter(regression_solver.RegressionParameters):
    "gaussian regression parameters"
    alpha: float = 1e-10
    n_restarts_optimizer: int = 0
    random_state: int | None = None


class GaussianRegressor(regression_solver.AbstractRegressor):
    "sklearn based Gaussian Process Regressor that integrates regressor and scaler"

    def __init__(self, parameters: GaussianRegressionParameter| regression_solver.RegressionParameters,
                 kernel: Kernel | None = None):
        "Initialize the solver."
        parameters = GaussianRegressionParameter(**attrs.asdict(parameters))
        self.kernel = self._get_kernel(kernel)
        super().__init__(parameters)

    def clone(self, parameters: GaussianRegressionParameter| regression_solver.RegressionParameters) -> 'GaussianRegressor':
        "Clone the regressor."
        return GaussianRegressor(parameters, self.kernel)


    def _get_scalers(self) -> tuple[LogComboScaler, LogComboScaler]:
        "Return the input and output scalers, Use StandardScaler for GPR"
        return LogComboScaler(StandardScaler()), LogComboScaler(StandardScaler())


    def _get_regressor(self) -> sklearn.gaussian_process.GaussianProcessRegressor:
        "Return the regressor."
        return sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.parameters.n_restarts_optimizer,
            alpha=self.parameters.alpha,
            random_state=self.parameters.random_state)


    def _get_kernel(self, kernel: Kernel | None) -> Kernel:
        "Return the kernel."
        if kernel is None:
            return ConstantKernel() * Matern(length_scale=1.0, nu=2.5) + WhiteKernel()
        return kernel


    def _fit(self):
        "fit the regressor"
        self.regressor.fit(self.data_regression.scaled_split.train_input,
                           self.data_regression.scaled_split.train_output)
        self.fitted = True
        self.update_error_metrics()


    def predict_uncertainty(self, input_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        "predict the output data along with the uncertainty"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted.")

        input_data_scaled = self.data_regression.input_scaler.transform(input_data)

        output_data_scaled, output_std_scaled = self.regressor.predict(input_data_scaled,
                                                                       return_std=True)

        output_data_scaled = pd.DataFrame(output_data_scaled,
                                          columns=self.parameters.output_data.columns)
        output_std_scaled = pd.DataFrame(output_std_scaled,
                                            columns=self.parameters.output_data.columns)

        output_data = self.data_regression.output_scaler.inverse_transform(output_data_scaled)

        output_std = self.data_regression.output_scaler.inverse_transform(output_std_scaled + output_data_scaled) - output_data

        return output_data, output_std

    def clone_tune(self, trial:optuna.trial.Trial) -> 'GaussianRegressor':
        "Create a new regressor with the same parameters and scalers as the original."
        raise NotImplementedError("Tuning is not implemented for the gaussian regressor")

    def auto_tune(self, settings:regression_solver.OptunaParameters) -> None:
        "Automatically tune the regressor."
        raise NotImplementedError("Tuning is not implemented for the gaussian regressor")

    def get_tuned_parameters(self) -> dict:
        raise NotImplementedError("Tuning is not implemented for the gaussian regressor")
