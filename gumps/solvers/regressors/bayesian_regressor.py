# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bayesian regression solver with input and output scaling."""

import attrs
import sklearn.linear_model
import sklearn.multioutput
import sklearn.preprocessing
import optuna

from gumps.solvers.regressors.polynomial_regressor import PolynomialRegressor, PolynomialRegressionParameters
import gumps.solvers.regressors.regression_solver as regression_solver


@attrs.define(kw_only=True)
class BayesianRegressionParameters(PolynomialRegressionParameters):
    "Parameters for the BayesianRegressor"
    n_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    alpha_init: float|None = None
    lambda_init: float|None = None
    compute_score: bool = False
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


class BayesianRegressor(PolynomialRegressor):
    "Bayesian regression solver with input and output scaling."

    #Allows proper typechecking
    def __init__(self, parameters:BayesianRegressionParameters):
        self.parameters: BayesianRegressionParameters
        super().__init__(parameters)

    def clone(self, parameters:BayesianRegressionParameters) -> 'BayesianRegressor':
        "Clone the solver."
        return BayesianRegressor(parameters)

    def _get_regressor(self) -> sklearn.multioutput.MultiOutputRegressor:
        "Return the regressor."
        regressor = sklearn.linear_model.BayesianRidge(n_iter=self.parameters.n_iter,
                                                  tol=self.parameters.tol,
                                                  alpha_1=self.parameters.alpha_1,
                                                  alpha_2=self.parameters.alpha_2,
                                                  lambda_1=self.parameters.lambda_1,
                                                  lambda_2=self.parameters.lambda_2,
                                                  alpha_init=self.parameters.alpha_init,
                                                  lambda_init=self.parameters.lambda_init,
                                                  compute_score=self.parameters.compute_score,
                                                  fit_intercept=self.parameters.fit_intercept,
                                                  copy_X=self.parameters.copy_X,
                                                  verbose=self.parameters.verbose)
        return sklearn.multioutput.MultiOutputRegressor(regressor)

    def auto_tune(self, settings:regression_solver.OptunaParameters) -> None:
        "Automatically tune the regressor."
        raise NotImplementedError("Tuning is not implemented for the bayesian regressor")

    def get_tuned_parameters(self) -> dict:
        "return the current value of parameters that can be tuned"
        raise NotImplementedError("Tuning is not implemented for the bayesian regressor")
