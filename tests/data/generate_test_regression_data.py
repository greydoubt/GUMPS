# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#generate saved joblib files regression tests, this is needed to handle sklearn warnings and errors related to joblib

from pathlib import Path
import pandas as pd
import numpy as np

import gumps.solvers.regressors.bayesian_regressor as bayesian_regressor
import gumps.solvers.regressors.gaussian_regressor as gaussian_regressor
import gumps.solvers.regressors.lasso_regressor as lasso_regressor
import gumps.solvers.regressors.lassocv_regressor as lassocv_regressor
import gumps.solvers.regressors.linear_regressor as linear_regressor
import gumps.solvers.regressors.mlp_regressor as mlp_regressor
import gumps.solvers.regressors.polynomial_regressor as polynomial_regressor
import gumps.solvers.regressors.pytorch_regressor as pytorch_regressor
import gumps.solvers.regressors.quadratic_regressor as quadratic_regressor
import gumps.solvers.regressors.random_forest_regressor as random_forest_regressor
import gumps.solvers.regressors.trivial_regressor as trivial_regressor
import gumps.solvers.regressors.regression_solver as regression_solver

import gumps.solvers.pca as pca
import gumps.solvers.kernel_pca as kernel_pca


def generate_test_regression_data(num_points=100) -> tuple[pd.DataFrame, pd.DataFrame]:
    "generate test data for a regression test"
    input_data = pd.DataFrame({"x1": np.linspace(0, 1, num_points),
                               "x2": np.linspace(1, 2, num_points),
                               "x3": np.linspace(2, 3, num_points)})
    output_data = pd.DataFrame({"y1": np.linspace(0, 1, num_points),
                                "y2": np.linspace(1, 2, num_points)})

    #add noise (this is not needed for the bayesian regressor)
    output_data *= np.random.normal(1, 0.01, output_data.shape)

    return input_data, output_data

def bayesian_save(input_data, output_data, base_directory):
    "create and train a bayesian regressor and then save it"
    directory = base_directory / "regressor" / "bayesian"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = bayesian_regressor.BayesianRegressionParameters(input_data=input_data,
                                     output_data=output_data,
                                     order = 1)
    regressor = bayesian_regressor.BayesianRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def gaussian_save(input_data, output_data, base_directory):
    "create and train a gaussian regressor and then save it"
    directory = base_directory / "regressor" / "gaussian"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = gaussian_regressor.GaussianRegressionParameter(input_data=input_data,
                                     output_data=output_data)
    regressor = gaussian_regressor.GaussianRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def lasso_save(input_data, output_data, base_directory):
    "create and train a lasso regressor and then save it"
    directory = base_directory / "regressor" / "lasso"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = lasso_regressor.LassoRegressionParameters(input_data=input_data,
                                     output_data=output_data,
                                     order=1)
    regressor = lasso_regressor.LassoRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def lassocv_save(input_data, output_data, base_directory):
    "create and train a lasso regressor and then save it"
    directory = base_directory / "regressor" / "lassocv"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = lassocv_regressor.LassoCVRegressionParameters(input_data=input_data,
                                     output_data=output_data,
                                     order=1)
    regressor = lassocv_regressor.LassoCVRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def linear_save(input_data, output_data, base_directory):
    "create and train a linear regressor and then save it"
    directory = base_directory / "regressor" / "linear"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = linear_regressor.LinearRegressionParameters(input_data=input_data,
                                     output_data=output_data)
    regressor = linear_regressor.LinearRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def mlp_save(input_data, output_data, base_directory):
    "create and train a mlp regressor and then save it"
    directory = base_directory / "regressor" / "mlp"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = mlp_regressor.MultiLayerPerceptronRegressionParameters(input_data=input_data,
                                     output_data=output_data)
    regressor = mlp_regressor.MultiLayerPerceptronRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def polynomial_save(input_data, output_data, base_directory):
    "create and train a polynomial regressor and then save it"
    directory = base_directory / "regressor" / "polynomial"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = polynomial_regressor.PolynomialRegressionParameters(input_data=input_data,
                                     output_data=output_data,
                                     order=2)
    regressor = polynomial_regressor.PolynomialRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def torch_mlp_save(input_data, output_data, base_directory):
    "create and train a torch mlp regressor and then save it"
    directory = base_directory / "regressor" / "torch_mlp"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = pytorch_regressor.TorchMultiLayerPerceptronRegressionParameters(input_data=input_data,
                                     output_data=output_data)
    regressor = pytorch_regressor.TorchMultiLayerPerceptronRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def quadratic_save(input_data, output_data, base_directory):
    "create and train a quadratic regressor and then save it"
    directory = base_directory / "regressor" / "quadratic"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = quadratic_regressor.QuadraticRegressionParameter(input_data=input_data,
                                     output_data=output_data)
    regressor = quadratic_regressor.QuadraticRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def random_forest_save(input_data, output_data, base_directory):
    "create and train a random forest regressor and then save it"
    directory = base_directory / "regressor" / "random_forest"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = random_forest_regressor.RandomForestRegressionParameters(input_data=input_data,
                                     output_data=output_data)
    regressor = random_forest_regressor.RandomForestRegressor(parameters)
    regressor.fit()
    regressor.save(directory)

def trivial_save(input_data, output_data, base_directory):
    "create and train a trivial regressor and then save it"
    directory = base_directory / "regressor" / "trivial"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = regression_solver.RegressionParameters(input_data=input_data,
                                     output_data=output_data)
    regressor = trivial_regressor.TrivialLinearRegressor(parameters)
    regressor.fit()
    regressor.save(directory)


def pca_save(input_data, base_directory):
    "create and train a pca solver and then save it"
    directory = base_directory / "solver" / "pca"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = pca.PCASettings(input_data=input_data)

    solver = pca.PCA(parameters)
    solver.fit()
    solver.save(directory)

def kernel_pca_save(input_data, base_directory):
    "create and train a kernel pca solver and then save it"
    directory = base_directory / "solver" / "kernel_pca"
    directory.mkdir(parents=True, exist_ok=True)

    parameters = kernel_pca.KernelPCASettings(input_data=input_data)

    solver = kernel_pca.KernelPCA(parameters)
    solver.fit()
    solver.save(directory)

def main():
    "generate the test data"
    base_directory = Path(__file__).parent
    input_data, output_data = generate_test_regression_data()

    #save the data for all regressors
    bayesian_save(input_data, output_data, base_directory)
    gaussian_save(input_data, output_data, base_directory)
    lasso_save(input_data, output_data, base_directory)
    lassocv_save(input_data, output_data, base_directory)
    linear_save(input_data, output_data, base_directory)
    mlp_save(input_data, output_data, base_directory)
    polynomial_save(input_data, output_data, base_directory)
    torch_mlp_save(input_data, output_data, base_directory)
    quadratic_save(input_data, output_data, base_directory)
    random_forest_save(input_data, output_data, base_directory)
    trivial_save(input_data, output_data, base_directory)

    # save the data for all solvers
    pca_save(input_data, base_directory)
    kernel_pca_save(input_data, base_directory)

if __name__ == "__main__":
    main()