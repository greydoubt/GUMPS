# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a linear regressor with automatic input and output scaling."""

import attrs
import optuna

from gumps.solvers.regressors.polynomial_regressor import PolynomialRegressor

from gumps.solvers.regressors.polynomial_regressor import PolynomialRegressionParameters
import gumps.solvers.regressors.regression_solver as regression_solver


@attrs.define(kw_only=True)
class LinearRegressionParameters(PolynomialRegressionParameters):
    "linear regression parameters"
    order: int = attrs.field(default=1)

    @order.validator
    def check(self, attribute, value):
        "valid that the order is 1"
        if value != 1:
            raise ValueError(f"Linear regression only support order = 1 {value}")

class LinearRegressor(PolynomialRegressor):
    "linear regressor with automatic input and output scaling"

    def __init__(self, parameters: LinearRegressionParameters):
        "initialize the linear regressor, this is here for type checking"
        super().__init__(parameters)

    def clone(self, parameters: LinearRegressionParameters) -> 'LinearRegressor':
        "Clone the regressor"
        return LinearRegressor(parameters)

    def auto_tune(self, settings:regression_solver.OptunaParameters) -> None:
        "Automatically tune the regressor."
        raise NotImplementedError("Tuning is not implemented for the linear regressor")

    def get_tuned_parameters(self) -> dict:
        raise NotImplementedError("Tuning is not implemented for the linear regressor")
