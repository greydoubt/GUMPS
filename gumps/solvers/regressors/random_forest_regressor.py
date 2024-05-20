# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a random forest regressor with input and output scaling"""


import attrs
import sklearn.ensemble
import optuna

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.minmax_scaler import MinMaxScaler
import gumps.solvers.regressors.regression_solver as regression_solver


@attrs.define(kw_only=True)
class RandomForestRegressionParameters(regression_solver.RegressionParameters):
    "random forest regression parameters"
    n_estimators: int = 100
    criterion: str = "squared_error"
    max_depth: int | None = None
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: str | int | float | None = 1.0
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int | None = None
    random_state: int | None = None
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: int | float | None = None


class RandomForestRegressor(regression_solver.AbstractRegressor):
    "This is a random forest regressor with input and output scaling"

    def __init__(self, parameters: RandomForestRegressionParameters| regression_solver.RegressionParameters):
        "Initialize the solver."
        forest_parameters = RandomForestRegressionParameters(**attrs.asdict(parameters))
        super().__init__(forest_parameters)

    def _get_scalers(self) -> tuple[LogComboScaler, LogComboScaler]:
        "Return the input and output scalers, Use StandardScaler for GPR"
        return LogComboScaler(MinMaxScaler()), LogComboScaler(MinMaxScaler())


    def _get_regressor(self) -> sklearn.ensemble.RandomForestRegressor:
        "Return the regressor."
        return sklearn.ensemble.RandomForestRegressor(n_estimators=self.parameters.n_estimators,
                                                      criterion=self.parameters.criterion,
                                                      max_depth=self.parameters.max_depth,
                                                      min_samples_split=self.parameters.min_samples_split,
                                                      min_samples_leaf=self.parameters.min_samples_leaf,
                                                      min_weight_fraction_leaf=self.parameters.min_weight_fraction_leaf,
                                                      max_features=self.parameters.max_features,
                                                      max_leaf_nodes=self.parameters.max_leaf_nodes,
                                                      min_impurity_decrease=self.parameters.min_impurity_decrease,
                                                      bootstrap=self.parameters.bootstrap,
                                                      oob_score=self.parameters.oob_score,
                                                      n_jobs=self.parameters.n_jobs,
                                                      random_state=self.parameters.random_state,
                                                      verbose=self.parameters.verbose,
                                                      warm_start=self.parameters.warm_start,
                                                      ccp_alpha=self.parameters.ccp_alpha,
                                                      max_samples=self.parameters.max_samples)


    def _fit(self):
        "fit the regressor"
        self.regressor.fit(self.data_regression.scaled_split.train_input,
                           self.data_regression.scaled_split.train_output)
        self.fitted = True
        self.update_error_metrics()

    def clone(self, parameters: RandomForestRegressionParameters| regression_solver.RegressionParameters) -> 'RandomForestRegressor':
        "Clone the regressor with new parameters."
        return RandomForestRegressor(parameters)

    def clone_tune(self, trial:optuna.trial.Trial) -> 'RandomForestRegressor':
        "Create a new regressor with the same parameters and scalers as the original."
        raise NotImplementedError("Tuning is not implemented for the random forest regressor")

    def auto_tune(self, settings:regression_solver.OptunaParameters) -> None:
        "Automatically tune the regressor."
        raise NotImplementedError("Tuning is not implemented for the random forest regressor")

    def get_tuned_parameters(self) -> dict:
        "Return the tuned parameters."
        raise NotImplementedError("Tuning is not implemented for the random forest regressor")
