# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a multioutput Lasso regressor with cross-validation and automatic input and output scaling"""

import attrs
import sklearn.linear_model
import sklearn.preprocessing
import optuna

from gumps.solvers.regressors.polynomial_regressor import PolynomialRegressor, PolynomialRegressionParameters
import gumps.solvers.regressors.regression_solver as regression_solver


@attrs.define(kw_only=True)
class LassoCVRegressionParameters(PolynomialRegressionParameters):
    "Parameters for the LassoCVRegressor"
    eps: float = 1e-3
    n_alphas: int = 100
    alphas: list[float]|None = None
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    copy_X: bool = True
    cv: int|None = None
    verbose: bool = False
    n_jobs: int|None = None
    random_state: int|None = None
    selection: str = 'cyclic'


class LassoCVRegressor(PolynomialRegressor):
    "Implement a multioutput Lasso regressor with cross-validation and automatic input and output scaling"

    #Allows proper typechecking
    def __init__(self, parameters:LassoCVRegressionParameters):
        self.parameters: LassoCVRegressionParameters
        super().__init__(parameters)
        self.poly = self._get_poly()

    def clone(self, parameters:LassoCVRegressionParameters) -> 'LassoCVRegressor':
        "Clone the solver."
        return LassoCVRegressor(parameters)

    def _get_regressor(self) -> sklearn.linear_model.MultiTaskLassoCV:
        "Return the regressor."
        return sklearn.linear_model.MultiTaskLassoCV(eps=self.parameters.eps,
                                            n_alphas=self.parameters.n_alphas,
                                            alphas=self.parameters.alphas,
                                            fit_intercept=self.parameters.fit_intercept,
                                            max_iter=self.parameters.max_iter,
                                            tol=self.parameters.tol,
                                            copy_X=self.parameters.copy_X,
                                            cv=self.parameters.cv,
                                            verbose=self.parameters.verbose,
                                            n_jobs=self.parameters.n_jobs,
                                            random_state=self.parameters.random_state,
                                            selection=self.parameters.selection)
