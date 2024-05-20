# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import matplotlib
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from gumps.graph.regressor import (FigureData, RegressorPlot,
                                   RegressorPlotParameters)
from gumps.solvers.regressors.regression_solver import RegressionParameters
from gumps.solvers.regressors.trivial_regressor import TrivialLinearRegressor

matplotlib.use('agg')

import matplotlib.pyplot as plt


def generate_regression_parameters(count: int = 5):
    "generate regression parameters"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
    return RegressionParameters(input_data=input_data, output_data=output_data)

def generate_regressor_plot_parameters(count: int = 5):
    "generate regressor plot parameters"
    input_data = pd.DataFrame({"x1": np.linspace(1, 5, count), 'x2': np.linspace(1, 2, count)})
    output_data = pd.DataFrame({"y1": np.linspace(2, 3, count), 'y2': np.linspace(1, 4, count)})

    reg_data = RegressionParameters(input_data=input_data, output_data=output_data)

    start = input_data.mean()
    lower_bound = input_data.min()
    upper_bound = input_data.max()

    regressor = TrivialLinearRegressor(reg_data)

    return RegressorPlotParameters(regressor, start, lower_bound, upper_bound)

def generate_regressor_plot_parameters_1D(count: int = 5):
    "generate regressor plot parameters"
    input_data = pd.DataFrame({"x1": np.linspace(1, 5, count), 'x2': np.linspace(1, 2, count)})
    output_data = pd.DataFrame({"y1": np.linspace(2, 3, count)})

    reg_data = RegressionParameters(input_data=input_data, output_data=output_data)

    start = input_data.mean()
    lower_bound = input_data.min()
    upper_bound = input_data.max()

    regressor = TrivialLinearRegressor(reg_data)

    return RegressorPlotParameters(regressor, start, lower_bound, upper_bound)

class TestRegressorPlotParameters(unittest.TestCase):
    "test the regressor plot parameters dataclass"

    def test_create_parameters(self):
        "create the parameters for the regressor plot"
        start = pd.Series({'x':0.0, 'y':0.0})
        lower_bound = pd.Series({'x':-1.0, 'y':-1.0})
        upper_bound = pd.Series({'x':1.0, 'y':1.0})

        regressor = TrivialLinearRegressor(generate_regression_parameters())

        params = RegressorPlotParameters(regressor, start, lower_bound, upper_bound)

        self.assertEqual(params.regressor, regressor)
        pd.testing.assert_series_equal(params.start, start)
        pd.testing.assert_series_equal(params.lower_bound, lower_bound)
        pd.testing.assert_series_equal(params.upper_bound, upper_bound)
        self.assertEqual(params.points, 100)

class TestFigureData(unittest.TestCase):
    "test the figure data dataclass"

    def test_create_figure_data(self):
        "create the figure data"
        input_data = pd.DataFrame({"x": np.linspace(0, 1, 5)})
        output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
        predicted_output_data = pd.DataFrame({"y": np.linspace(0, 1, 5)})
        response_input_data = {'x': pd.DataFrame({"x": np.linspace(0, 1, 5)})}
        response_output_data = {'x': pd.DataFrame({"y": np.linspace(0, 1, 5)})}

        data = FigureData(input_data, output_data, predicted_output_data, response_input_data, response_output_data)

        pd.testing.assert_frame_equal(data.input_data, input_data)
        pd.testing.assert_frame_equal(data.output_data, output_data)
        pd.testing.assert_frame_equal(data.predicted_output_data, predicted_output_data)
        self.assertEqual(data.response_input_data, response_input_data)
        self.assertEqual(data.response_output_data, response_output_data)

    def test_get_response_data(self):
        "get the response data"
        input_data = pd.DataFrame({"x1": np.linspace(0, 1, 5), 'x2': np.linspace(1, 2, 5)})
        output_data = pd.DataFrame({"y1": np.linspace(0, 1, 5), 'y2': np.linspace(0, 1, 5)})

        predicted_output_data = pd.DataFrame({"y1": np.linspace(0, 1, 5), 'y2': np.linspace(0, 1, 5)})

        response_input_data = {'x1': pd.DataFrame({"x1": np.linspace(0, 1, 5), 'x2': np.repeat(1.5, 5)}),
                                'x2': pd.DataFrame({"x1": np.repeat(0.5, 5), 'x2': np.linspace(1, 2, 5)})}

        response_output_data = {'x1': pd.DataFrame({"y1": np.linspace(0, 1, 5), 'y2': np.linspace(1, 2, 5)}),
                                 'x2': pd.DataFrame({"y1": np.linspace(2, 3, 5), 'y2': np.linspace(3, 4, 5)})}

        data = FigureData(input_data, output_data, predicted_output_data, response_input_data, response_output_data)

        response_data = data.get_response_data("y1")

        self.assertEqual(list(response_data.keys()), ['x1', 'x2'])
        pd.testing.assert_frame_equal(response_data['x1'], pd.DataFrame({'x': response_input_data['x1']['x1'],'y': response_output_data['x1']['y1']}))
        pd.testing.assert_frame_equal(response_data['x2'], pd.DataFrame({'x': response_input_data['x2']['x2'],'y': response_output_data['x2']['y1']}))

class TestRegressor(unittest.TestCase):
    "test the gaussian process regressor"

    def test_init(self):
        "test the initialization of the regressor"
        params = generate_regressor_plot_parameters(10)

        plot = RegressorPlot(params)

        self.assertEqual(plot.params, params)

    def test_get_plot(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        fig = plot.get_plot()

        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_get_scaled_plot(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        fig = plot.get_plot_scaled()

        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_save_plot(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            plot.save_plot(temp_dir / "test.png")
            self.assertTrue((temp_dir / "test.png").exists())

    def test_save_scaled_plot(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            plot.save_scaled_plot(temp_dir / "test.png")
            self.assertTrue((temp_dir / "test.png").exists())

    def test_plot(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        plot.plot()

    def test_plot_1d(self):
        "test plotting a 1d regressor"
        params = generate_regressor_plot_parameters_1D(10)
        params.regressor.fit()

        plot = RegressorPlot(params)
        plot.plot()

    def test_plot_scaled(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        plot.plot_scaled()

    def test_get_subplot_size(self):
        count = 5
        input_data = pd.DataFrame({"x1": np.linspace(1, 5, count), 'x2': np.linspace(1, 2, count)})
        output_data = pd.DataFrame({"y1": np.linspace(2, 3, count), 'y2': np.linspace(1, 4, count),
                                    'y3': np.linspace(1, 4, count), 'y4': np.linspace(1, 4, count)})

        reg_data = RegressionParameters(input_data=input_data, output_data=output_data)

        start = input_data.mean()
        lower_bound = input_data.min()
        upper_bound = input_data.max()

        regressor = TrivialLinearRegressor(reg_data)

        params =  RegressorPlotParameters(regressor, start, lower_bound, upper_bound)

        plot = RegressorPlot(params)

        self.assertDictEqual(plot._get_subplot_size(), {'nrows': 4, 'ncols': 3})

    def test_create_figure(self):
        "test that the figure is created without an exception"
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        subplot_size = plot._get_subplot_size()
        _, axes = plt.subplots(**subplot_size)

        input_data, output_data, predicted_output_data = plot._get_prediction_data()
        response_input_data, response_output_data = plot._get_response_data()
        data = FigureData(input_data, output_data, predicted_output_data, response_input_data, response_output_data)

        fig = plot._create_figure(data)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_get_prediction_data(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        input_data, output_data, predicted_output_data = plot._get_prediction_data()

        norm = np.linalg.norm((output_data - predicted_output_data).values)

        self.assertLess(norm, 1e-14)
        pd.testing.assert_frame_equal(input_data, params.regressor.parameters.input_data)
        pd.testing.assert_frame_equal(output_data, params.regressor.parameters.output_data)
        pd.testing.assert_frame_equal(predicted_output_data, params.regressor.parameters.output_data)


    def test_get_prediction_data_scaled(self):
        params = generate_regressor_plot_parameters(100)
        params.regressor.fit()

        plot = RegressorPlot(params)

        input_data, output_data, predicted_output_data = plot._get_prediction_data_scaled()

        pd.testing.assert_series_equal(input_data.min(), pd.Series({'x1': 0.0, 'x2': 0.0}))
        pd.testing.assert_series_equal(input_data.max(), pd.Series({'x1': 1.0, 'x2': 1.0}))
        pd.testing.assert_series_equal(output_data.min(), pd.Series({'y1': 0.0, 'y2': 0.0}))
        pd.testing.assert_series_equal(output_data.max(), pd.Series({'y1': 1.0, 'y2': 1.0}))
        pd.testing.assert_series_equal(predicted_output_data.min(), pd.Series({'y1': 0.0, 'y2': 0.0}))
        pd.testing.assert_series_equal(predicted_output_data.max(), pd.Series({'y1': 1.0, 'y2': 1.0}))

        norm = np.linalg.norm((output_data - predicted_output_data).values)

        self.assertLess(norm, 1e-14)


    def test_scale_input(self):
        params = generate_regressor_plot_parameters(100)

        plot = RegressorPlot(params)

        input_data_scaled = plot._scale_input(plot.params.regressor.parameters.input_data)

        pd.testing.assert_series_equal(input_data_scaled.min(), pd.Series({'x1': 0.0, 'x2': 0.0}))
        pd.testing.assert_series_equal(input_data_scaled.max(), pd.Series({'x1': 1.0, 'x2': 1.0}))

    def test_scale_output(self):
        params = generate_regressor_plot_parameters(100)

        plot = RegressorPlot(params)

        output_data_scaled = plot._scale_output(plot.params.regressor.parameters.output_data)

        pd.testing.assert_series_equal(output_data_scaled.min(), pd.Series({'y1': 0.0, 'y2': 0.0}))
        pd.testing.assert_series_equal(output_data_scaled.max(), pd.Series({'y1': 1.0, 'y2': 1.0}))

    def test_get_response_data_scaled(self):
        "test the response data generation, use a simpler synthetic problem so that the output can be verified"
        input_data = pd.DataFrame({"x1": np.linspace(0, 1, 11)})
        output_data = pd.DataFrame({"y1": np.linspace(0, 1, 11)})

        reg_data = RegressionParameters(input_data=input_data, output_data=output_data)

        start = input_data.mean()
        lower_bound = input_data.min()
        upper_bound = input_data.max()

        regressor = TrivialLinearRegressor(reg_data)

        params =  RegressorPlotParameters(regressor, start, lower_bound, upper_bound)
        params.regressor.fit()

        plot = RegressorPlot(params)

        response_input_data, response_output_data = plot._get_response_data_scaled()

        self.assertEqual(list(response_input_data.keys()), ['x1',])
        self.assertEqual(list(response_output_data.keys()), ['x1',])

        pd.testing.assert_series_equal(response_input_data['x1'].min(), pd.Series({'x1': 0.0}))
        pd.testing.assert_series_equal(response_input_data['x1'].max(), pd.Series({'x1': 1.0}))

        pd.testing.assert_series_equal(response_output_data['x1'].min(), pd.Series({'y1': 0.0}))
        pd.testing.assert_series_equal(response_output_data['x1'].max(), pd.Series({'y1': 1.0}))

    def test_get_response_data(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        response_input_data, response_output_data = plot._get_response_data()

        self.assertEqual(list(response_input_data.keys()), ['x1', 'x2'])
        self.assertEqual(list(response_output_data.keys()), ['x1', 'x2'])

    def test_get_response_data_for_label(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        response_input_data, response_output_data = plot._get_response_data_for_label("x1")

        self.assertIsInstance(response_input_data, pd.DataFrame)
        self.assertIsInstance(response_output_data, pd.DataFrame)
        self.assertAlmostEqual(response_input_data['x1'].min(), 1.0)
        self.assertAlmostEqual(response_input_data['x1'].max(), 5.0)
        self.assertAlmostEqual(response_input_data['x2'].min(), 1.5)
        self.assertAlmostEqual(response_input_data['x2'].max(), 1.5)

    def test_create_prediction_subplot(self):
        "test that the subplot can be created without exception"
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        _, axes = plt.subplots(nrows=1, ncols=1)

        _, output_data, predicted_output_data = plot._get_prediction_data()

        plot._create_prediction_subplot(axes, response_label="y1",
                                        output_data = output_data,
                                        predicted_output_data = predicted_output_data)

    def test_create_response_subplot(self):
        params = generate_regressor_plot_parameters(10)
        params.regressor.fit()

        plot = RegressorPlot(params)

        subplot_size = plot._get_subplot_size()
        _, axes = plt.subplots(**subplot_size)

        input_data, output_data, predicted_output_data = plot._get_prediction_data()
        response_input_data, response_output_data = plot._get_response_data()
        data = FigureData(input_data, output_data, predicted_output_data, response_input_data, response_output_data)

        plot._create_response_subplot(axes[0,1:], response_label="y1", response_data=data.get_response_data("y1"))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRegressorPlotParameters)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestFigureData)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestRegressor)
    unittest.TextTestRunner(verbosity=2).run(suite)