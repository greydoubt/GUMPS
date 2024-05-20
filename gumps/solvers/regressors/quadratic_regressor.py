# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implements a quadratic regressor with automatic scaling of the input and output data."""

import attrs
import optuna

from gumps.solvers.regressors.polynomial_regressor import PolynomialRegressor

from gumps.solvers.regressors.polynomial_regressor import PolynomialRegressionParameters
import gumps.solvers.regressors.regression_solver as regression_solver


@attrs.define(kw_only=True)
class QuadraticRegressionParameter(PolynomialRegressionParameters):
    "Quadratic regression parameters"
    order: int = attrs.field(default = 2)

    @order.validator
    def check(self, attribute, value):
        "validate that the order is 2"
        if value != 2:
            raise ValueError(f"Quadratic regression only support order = 2 {attribute} = {value}")

class QuadraticRegressor(PolynomialRegressor):
    "Quadratic regressor with automatic input and output scaling"

    def __init__(self, parameters: QuadraticRegressionParameter):
        "initialize the quadratic regressor, this is here for type checking"
        super().__init__(parameters)

    def clone(self, parameters: QuadraticRegressionParameter) -> 'QuadraticRegressor':
        "Clone the regressor"
        return QuadraticRegressor(parameters)

    def auto_tune(self) -> None:
        "Automatically tune the regressor."
        super().auto_tune(max_order=2)
