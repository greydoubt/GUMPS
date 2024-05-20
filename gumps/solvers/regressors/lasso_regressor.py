# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a lasso regressor with input and output scaling."""

import attrs
import sklearn.linear_model
import sklearn.multioutput
import sklearn.preprocessing
import optuna

from gumps.solvers.regressors.polynomial_regressor import PolynomialRegressor, PolynomialRegressionParameters
import gumps.solvers.regressors.regression_solver as regression_solver


@attrs.define(kw_only=True)
class LassoRegressionParameters(PolynomialRegressionParameters):
    "Parameters for the LassoRegressor"
    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    random_state: int|None = None
    selection: str = 'cyclic'


class LassoRegressor(PolynomialRegressor):
    "A regressor that uses Lasso regression."

    #Allows proper typechecking
    def __init__(self, parameters:LassoRegressionParameters):
        self.parameters: LassoRegressionParameters
        super().__init__(parameters)
        self.poly = self._get_poly()

    def clone(self, parameters:LassoRegressionParameters) -> 'LassoRegressor':
        "Clone the solver."
        return LassoRegressor(parameters)

    def _get_regressor(self) -> sklearn.multioutput.MultiOutputRegressor:
        "Return the regressor."
        regressor = sklearn.linear_model.Lasso(alpha=self.parameters.alpha,
                                          fit_intercept=self.parameters.fit_intercept,
                                          copy_X=self.parameters.copy_X,
                                          max_iter=self.parameters.max_iter,
                                          tol=self.parameters.tol,
                                          warm_start=self.parameters.warm_start,
                                          positive=self.parameters.positive,
                                          random_state=self.parameters.random_state,
                                          selection=self.parameters.selection)
        return sklearn.multioutput.MultiOutputRegressor(regressor)

    def auto_tune(self, settings:regression_solver.OptunaParameters) -> None:
        "Automatically tune the regressor."
        raise NotImplementedError("Tuning is not implemented for the lasso regressor")

    def get_tuned_parameters(self) -> dict:
        raise NotImplementedError("Tuning is not implemented for the lasso regressor")
