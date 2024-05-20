# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a multi-layer perceptron regressor with input and output scaling."""

import attrs
import sklearn.neural_network
import optuna

import gumps.solvers.regressors.regression_solver as regression_solver
from gumps.scalers.standard_scaler import StandardScaler
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.log_combo_scaler import LogComboScaler


@attrs.define(kw_only=True)
class MultiLayerPerceptronRegressionParameters(regression_solver.RegressionParameters):
    "Parameters for the multi-layer perceptron regressor."
    hidden_layer_sizes: tuple[int, ...] = (100,)
    alpha: float = 0.0001
    batch_size: str = "auto"
    learning_rate: str = "constant"
    learning_rate_init: float = 0.001
    max_iter: int = 400
    shuffle: bool = True
    random_state: int | None = None
    tol: float = 0.0001
    verbose: bool = False
    early_stopping: bool = True
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-08
    n_iter_no_change: int = 10
    validation_fraction: float = attrs.field(default=0.1)
    @validation_fraction.validator
    def _check_validation_fraction(self, _, value):
        if not (0 < value < 1):
            raise ValueError(f"validation_fraction must be between 0 and 1, not {value}")

class MultiLayerPerceptronRegressor(regression_solver.AbstractRegressor):
    "Implement a multi-layer perceptron regressor with input and output scaling."

    def __init__(self, parameters: MultiLayerPerceptronRegressionParameters):
        "Initialize the solver."
        self.parameters: MultiLayerPerceptronRegressionParameters
        super().__init__(parameters)

    def clone(self, parameters: MultiLayerPerceptronRegressionParameters) -> 'MultiLayerPerceptronRegressor':
        "Clone the solver."
        return MultiLayerPerceptronRegressor(parameters)


    def _get_scalers(self) -> tuple[LogComboScaler, LogComboScaler]:
        """Return the input and output scalers, For regression neural networks
        use a standard scaler for the input and a min max scaler for the output."""
        return LogComboScaler(StandardScaler()), LogComboScaler(MinMaxScaler())


    def _get_regressor(self) -> sklearn.neural_network.MLPRegressor:
        "Return the regressor."
        return sklearn.neural_network.MLPRegressor(hidden_layer_sizes=self.parameters.hidden_layer_sizes,
                                                   alpha=self.parameters.alpha,
                                                   batch_size=self.parameters.batch_size,
                                                   learning_rate=self.parameters.learning_rate,
                                                   learning_rate_init=self.parameters.learning_rate_init,
                                                   max_iter=self.parameters.max_iter,
                                                   shuffle=self.parameters.shuffle,
                                                   random_state=self.parameters.random_state,
                                                   tol=self.parameters.tol,
                                                   verbose=self.parameters.verbose,
                                                   early_stopping=self.parameters.early_stopping,
                                                   validation_fraction=self.parameters.validation_fraction,
                                                   beta_1=self.parameters.beta_1,
                                                   beta_2=self.parameters.beta_2,
                                                   epsilon=self.parameters.epsilon,
                                                   n_iter_no_change=self.parameters.n_iter_no_change,
                                                   warm_start=True)


    def _fit(self):
        "fit the regressor"
        self.regressor.fit(self.data_regression.scaled_split.full_input,
                           self.data_regression.scaled_split.full_output)
        self.fitted = True
        self.update_error_metrics()

    def clone_tune(self, trial:optuna.trial.Trial) -> 'MultiLayerPerceptronRegressor':
        "Create a new regressor with the same parameters and scalers as the original."
        raise NotImplementedError("Tuning is not implemented for the multilayer perceptron regressor")

    def auto_tune(self, settings:regression_solver.OptunaParameters) -> None:
        "Automatically tune the regressor."
        raise NotImplementedError("Tuning is not implemented for the multilayer perceptron regressor")

    def get_tuned_parameters(self) -> dict:
        raise NotImplementedError("Tuning is not implemented for the multilayer perceptron regressor")
