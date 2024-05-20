# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from gumps.solvers.regressors.regression_loader import get_regressor

regressors = ["RandomForestRegressor",
              "QuadraticRegressor",
              "PolynomialRegressor",
              "LinearRegressor",
              "LassoRegressor",
              "LassoCVRegressor",
              "GaussianRegressor",
              "BayesianRegressor",
              "MultiLayerPerceptronRegressor",
              "TorchMultiLayerPerceptronRegressor",
              "XGBoostRegressor",
              "TrivialLinearRegressor",
              "EnsembleRegressor"]

@pytest.mark.parametrize("regressor", regressors)
def test_get_regressor(regressor:str):
    assert get_regressor(regressor).__name__ == regressor

def test_get_regressor_exception():
    with pytest.raises(ValueError):
        get_regressor("InvalidRegressor")