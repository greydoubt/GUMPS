# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"This file contains the logic for loading a regressor from a directory."

import pathlib

import gumps.solvers.regressors.regression_solver as regression_solver

def get_regressor(name:str) -> type[regression_solver.AbstractRegressor]:
    '''use a match statement to import and return the correct regressor'''
    match name:
        case 'RandomForestRegressor':
            import gumps.solvers.regressors.random_forest_regressor as random_forest_regressor
            return random_forest_regressor.RandomForestRegressor
        case 'QuadraticRegressor':
            import gumps.solvers.regressors.quadratic_regressor as quadratic_regressor
            return quadratic_regressor.QuadraticRegressor
        case 'PolynomialRegressor':
            import gumps.solvers.regressors.polynomial_regressor as polynomial_regressor
            return polynomial_regressor.PolynomialRegressor
        case 'LinearRegressor':
            import gumps.solvers.regressors.linear_regressor as linear_regressor
            return linear_regressor.LinearRegressor
        case 'LassoRegressor':
            import gumps.solvers.regressors.lasso_regressor as lasso_regressor
            return lasso_regressor.LassoRegressor
        case 'LassoCVRegressor':
            import gumps.solvers.regressors.lassocv_regressor as lassocv_regressor
            return lassocv_regressor.LassoCVRegressor
        case 'GaussianRegressor':
            import gumps.solvers.regressors.gaussian_regressor as gaussian_regressor
            return gaussian_regressor.GaussianRegressor
        case 'BayesianRegressor':
            import gumps.solvers.regressors.bayesian_regressor as bayesian_regressor
            return bayesian_regressor.BayesianRegressor
        case 'MultiLayerPerceptronRegressor':
            import gumps.solvers.regressors.mlp_regressor as mlp_regressor
            return mlp_regressor.MultiLayerPerceptronRegressor
        case 'TorchMultiLayerPerceptronRegressor':
            import gumps.solvers.regressors.pytorch_regressor as pytorch_regressor
            return pytorch_regressor.TorchMultiLayerPerceptronRegressor
        case 'XGBoostRegressor':
            import gumps.solvers.regressors.xgboost_regressor as xgboost_regressor
            return xgboost_regressor.XGBoostRegressor
        case 'TrivialLinearRegressor':
            import gumps.solvers.regressors.trivial_regressor as trivial_regressor
            return trivial_regressor.TrivialLinearRegressor
        case 'EnsembleRegressor':
            import gumps.solvers.regressors.ensemble_regressor as ensemble_regressor
            return ensemble_regressor.EnsembleRegressor
        case _:
            raise ValueError(f"Regressor {name} not found")

def load_regressor(path: pathlib.Path,
                   auto_rebuild: bool = True,
                   auto_resave: bool = True):
    '''Looks at the directory to determine which regressor to load'''
    with open(path / "regressor.txt", 'r', encoding='utf-8') as f:
        regressor_name = f.read()

    regressor_cls = get_regressor(regressor_name)
    return regressor_cls.load(path,
                              auto_rebuild=auto_rebuild,
                              auto_resave=auto_resave)