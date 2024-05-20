# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the gpr adaptive sampler"

import unittest
import copy

import numpy as np
import pandas as pd

from gumps.solvers.adaptive_solver import (AdaptiveSamplerParameters,
                                           GaussianRegressorAdaptiveSampler,
                                           WhatIf)
from gumps.solvers.regressors.gaussian_regressor import GaussianRegressor
from gumps.solvers.regressors.regression_solver import RegressionParameters

def model(input_data: pd.DataFrame) -> pd.DataFrame:
    "model"
    return pd.DataFrame({"y": 2 * input_data["x"]}) + np.random.normal(0, 0.005, input_data.shape)

def model_nd(input_data:pd.DataFrame) -> pd.DataFrame:
    "generate an nd model"
    output_data =  pd.DataFrame({"y1": 2 * input_data["x1"] + 3 * input_data["x2"] + 4 * input_data["x3"],
                                 "y2": 1 * input_data["x1"] + 2 * input_data["x2"] + 3 * input_data["x3"],})
    output_data = output_data + np.random.normal(0, 0.005, output_data.shape)
    return output_data

def generate_regression_parameters(count: int = 5):
    "generate regression parameters"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
    output_data = model(input_data)
    return RegressionParameters(input_data=input_data, output_data=output_data)

def generate_regression_parameters_nd(count: int = 5):
    "generate regression parameters"
    input_data = pd.DataFrame({"x1": np.linspace(0, 1, count),
                               'x2': np.linspace(1, 2, count),
                               'x3': np.linspace(2, 3, count),})
    output_data = model_nd(input_data)
    return RegressionParameters(input_data=input_data, output_data=output_data)

class TestAdaptiveSampler(unittest.TestCase):
    "test the adaptive sampler"

    def test_parameters(self):
        "test the parameters"
        params = AdaptiveSamplerParameters(points_to_add=5)
        self.assertEqual(params.points_to_add, 5)
        self.assertEqual(params.batch_size, 1)
        self.assertEqual(params.max_iterations, 100)
        self.assertEqual(params.population_size, 100)

    def test_init(self):
        "test the init method"
        params = AdaptiveSamplerParameters(points_to_add=5)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        self.assertIsInstance(sampler.regressor, GaussianRegressor)
        self.assertIsInstance(sampler.solver_settings, AdaptiveSamplerParameters)

    def test_has_next_true(self):
        "test that more points are needed"
        params = AdaptiveSamplerParameters(points_to_add=5)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)
        self.assertTrue(sampler.has_next())

    def test_has_next_false(self):
        "test that more points are not needed"
        params = AdaptiveSamplerParameters(points_to_add=0)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)
        self.assertFalse(sampler.has_next())

    def test_whatif(self):
        "test the whatif method"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=4,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        whatif = sampler.whatif()

        self.assertIsInstance(whatif, WhatIf)
        self.assertIsInstance(whatif.input_data, pd.DataFrame)
        self.assertIsInstance(whatif.output_data, pd.DataFrame)
        self.assertIsInstance(whatif.uncertainty, pd.DataFrame)
        self.assertEqual(whatif.input_data.shape, (4, 1))
        self.assertEqual(whatif.output_data.shape, (4, 1))
        self.assertEqual(whatif.uncertainty.shape, (4, 1))
        self.assertIsInstance(whatif.regressor, GaussianRegressor)

    def test_ask(self):
        "test the ask method"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=4,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        next_points = sampler.ask()

        #This line is checking if the right shape is returned
        next_points = pd.DataFrame(next_points, columns=regressor.parameters.input_data.columns)
        self.assertIsInstance(next_points, pd.DataFrame)
        self.assertEqual(next_points.shape, (4, 1))

    def test_ask_nd(self):
        "test the ask method in nd"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=4,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters_nd()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        next_points = sampler.ask()

        #This line is checking if the right shape is returned
        next_points = pd.DataFrame(next_points, columns=regressor.parameters.input_data.columns)
        self.assertIsInstance(next_points, pd.DataFrame)
        self.assertEqual(next_points.shape, (4, 3))

    def test_ask_batch_size_one(self):
        "test with a batch size of one"
        params = AdaptiveSamplerParameters(points_to_add=2,
                                           batch_size=1,
                                           population_size=10,
                                           max_iterations=10)
        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        next_points = sampler.ask()

        #This line is checking if the right shape is returned
        next_points = pd.DataFrame(next_points, columns=regressor.parameters.input_data.columns)
        self.assertIsInstance(next_points, pd.DataFrame)
        self.assertEqual(next_points.shape, (1, 1))

    def test_ask_batch_size_one_nd(self):
        "test with a batch size of one"
        params = AdaptiveSamplerParameters(points_to_add=2,
                                           batch_size=1,
                                           population_size=10,
                                           max_iterations=10)
        regression_parameters = generate_regression_parameters_nd()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        next_points = sampler.ask()

        #This line is checking if the right shape is returned
        next_points = pd.DataFrame(next_points, columns=regressor.parameters.input_data.columns)
        self.assertIsInstance(next_points, pd.DataFrame)
        self.assertEqual(next_points.shape, (1, 3))

    def test_ask_one(self):
        "test the ask one method returns a single point"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=1,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        regressor_copy = copy.deepcopy(sampler.regressor)

        next_point, next_point_expected_output, next_point_uncertainty = sampler._ask_one(regressor_copy)

        self.assertIsInstance(next_point, pd.Series)
        self.assertTrue(len(next_point) == 1)
        self.assertTrue(len(next_point_expected_output) == 1)
        self.assertTrue(next_point_uncertainty > 0)

    def test_ask_one_nd(self):
        "test the ask one method returns a single point"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=1,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters_nd()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        regressor_copy = copy.deepcopy(sampler.regressor)

        next_point, next_point_expected_output, next_point_uncertainty = sampler._ask_one(regressor_copy)

        self.assertIsInstance(next_point, pd.Series)
        self.assertTrue(len(next_point) == 3)
        self.assertTrue(len(next_point_expected_output) == 2)
        self.assertTrue(next_point_uncertainty > 0)

    def test_tell(self):
        "test the tell method"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=1,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        next_points = sampler.ask()
        next_points = pd.DataFrame(next_points, columns=regressor.parameters.input_data.columns)

        output_data = model(next_points)

        sampler.tell(output_data)
        self.assertEqual(sampler.points_to_add, 4)

    def test_tell_nd(self):
        "test the tell method"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=1,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters_nd()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        next_points = sampler.ask()
        next_points = pd.DataFrame(next_points, columns=regressor.parameters.input_data.columns)

        output_data = model_nd(next_points)

        sampler.tell(output_data)
        self.assertEqual(sampler.points_to_add, 4)

    def test_tell_exception_ask_before_tell(self):
        "test the tell method"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=1,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        with self.assertRaises(RuntimeError):
            loss = pd.DataFrame()
            sampler.tell(loss)


    def test_tell_exception_shape_wrong(self):
        "test the tell method"
        params = AdaptiveSamplerParameters(points_to_add=5,
                                           batch_size=4,
                                           population_size=10,
                                           max_iterations=10)

        regression_parameters = generate_regression_parameters()
        regressor = GaussianRegressor(regression_parameters)
        regressor.fit()

        sampler = GaussianRegressorAdaptiveSampler(regressor, params)

        next_points = sampler.ask()
        next_points = pd.DataFrame(next_points, columns=regressor.parameters.input_data.columns)

        output_data = model(next_points)

        with self.assertRaises(ValueError):
            sampler.tell(output_data.iloc[0:2])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveSampler)
    unittest.TextTestRunner(verbosity=2).run(suite)
