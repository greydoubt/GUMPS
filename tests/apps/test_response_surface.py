# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#Create tests for the response surface app

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gumps.apps.response_surface import ResponseSurface
from gumps.common.parallel import Parallel
from gumps.solvers.response_sampler import ResponseSamplerParameters
from gumps.studies.batch_sphere_study import BatchLineStudy
from gumps.solvers.simple_solver import SimpleSolver
from gumps.studies.batch_study import BatchStudyMultiProcess
from gumps.kernels.sphere_kernel import SphereKernelNanException
from gumps.studies.study import SimulationStudy

def exception_handler(function, x, e):
    return None

def get_total(frame:pd.DataFrame|None):
    if frame is None:
        return None
    return {'total': frame.total[0]}

class TestResponseSurface(unittest.TestCase):

    def setUp(self):
        "setup the test"
        self.parameters = ResponseSamplerParameters(
                        lower_bound=pd.Series({'x1': -2, 'x2': -2, 'x3': 0.5}),
                        upper_bound=pd.Series({'x1': 2, 'x2': 0, 'x3': 1.5}),
                        baseline=pd.Series({'x1': 0, 'x2': -1, 'x3': 1}),
                        points_1d=5,
                        points_2d_per_dimension=5)
        self.batch = BatchLineStudy(dict(self.parameters.baseline))
        self.directory = None
        self.processing_function = lambda x: x
        self.pre_processing_function = None

        self. app = ResponseSurface(parameters=self.parameters,
                             batch=self.batch,
                             parallel=Parallel(ordered=False),
                             directory=self.directory,
                             processing_function=self.processing_function,
                             pre_processing_function=self.pre_processing_function)


    def test_init(self):
        self.assertEqual(self.app.parameters, self.parameters)
        self.assertEqual(self.app.batch, self.batch)
        self.assertEqual(self.app.directory, self.directory)
        self.assertEqual(self.app.processing_function, self.processing_function)
        self.assertEqual(self.app.pre_processing_function, self.pre_processing_function)


    def test_run(self):
        self.app.run()

        self.assertIsInstance(self.app.solver.response, pd.DataFrame)
        self.assertIsInstance(self.app.solver.request, pd.DataFrame)


    def test_run_with_pre_processing_function(self):
        pre_processing_function = lambda x: x**2
        self.app.pre_processing_function = pre_processing_function

        self.app.run()

        diff = (self.app.solver.request**2 - self.app.parameters.baseline).sum(axis=1)
        max_value = diff.max()
        min_value = diff.min()

        self.assertIsInstance(self.app.solver.response, pd.DataFrame)
        self.assertIsInstance(self.app.solver.request, pd.DataFrame)
        self.assertAlmostEqual(max_value, self.app.solver.response['total'].max())
        self.assertAlmostEqual(min_value, self.app.solver.response['total'].min())



    def test_request_variables(self):
        "test that the request variables are correct"
        self.app.run()

        self.assertListEqual(self.app.request_variables(), ['x1', 'x2', 'x3'])


    def test_response_variables(self):
        "test that the response variables are correct"
        self.app.run()

        self.assertListEqual(self.app.response_variables(), ['d1', 'd2', 'd3', 'total'])


    def test_request(self):
        "test that the request is correct"
        self.app.run()

        pd.testing.assert_frame_equal(self.app.request(), self.app.solver.request)


    def test_resposne(self):
        "test that the response is correct"
        self.app.run()

        pd.testing.assert_frame_equal(self.app.response(), self.app.solver.response)


    def test_split_response(self):
        "test that the split response is correct"
        self.app.run()

        baseline, data_1d, data_2d = self.app.split_response()


        self.assertIsInstance(baseline, pd.DataFrame)
        self.assertIsInstance(data_1d, dict)
        self.assertIsInstance(data_2d, dict)


    def test_clean_nan_values_no_nan(self):
        df_no_nan = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = ResponseSurface.clean_nan_values(df_no_nan)
        pd.testing.assert_frame_equal(df_no_nan, result)

    def test_clean_nan_values_all_nan(self):
        df_all_nan = pd.DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]})
        with self.assertRaises(ValueError):
            ResponseSurface.clean_nan_values(df_all_nan)

    def test_clean_nan_values_one_nan(self):
        df_one_nan = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [4.0, 5.0, 6.0]})

        with self.assertLogs('gumps.apps.response_surface', level='WARNING') as cm:
            result = ResponseSurface.clean_nan_values(df_one_nan)

            correct = ['WARNING:gumps.apps.response_surface:NaN values found in data, removing rows [1]'
                ]
            self.assertEqual(cm.output, correct)
        expected = pd.DataFrame({'A': [1.0, 3.0], 'B': [4.0, 6.0]})
        pd.testing.assert_frame_equal(expected, result)


    def test_get_1d_plots(self):
        response_name = 'total'

        self.app.run()

        figures = list(self.app.get_1d_plots(response_name))

        self.assertEqual(len(figures), len(self.parameters.baseline))

        for key, fig in figures:
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(fig.axes[0].get_xlabel(), key)
            self.assertEqual(fig.axes[0].get_ylabel(), response_name)
            self.assertEqual(fig.axes[0].get_title(), f'{key} vs {response_name}')
            self.assertEqual(fig.axes[0].get_xlim(), (self.parameters.lower_bound[key], self.parameters.upper_bound[key]))
            self.assertEqual(fig.axes[0].get_legend().get_texts()[0].get_text(), response_name)
            plt.close(fig)


    def test_show_1d_plots(self):
        response_name = 'total'

        figures = [
            ('key1', MagicMock()),
            ('key2', MagicMock()),
            ('key3', MagicMock())
        ]

        self.app.get_1d_plots = MagicMock(return_value=figures)

        with unittest.mock.patch('matplotlib.pyplot.show'):
            self.app.show_1d_plots(response_name)

        self.assertEqual(self.app.get_1d_plots.call_count, 1)


    def test_save_1d_plots(self):
        #set poolsize to 1 due to issues with tests, generators, multiprocessing, and saving
        self.app.parallel.poolsize = 1

        self.app.run()

        response_name = 'total'

        with tempfile.TemporaryDirectory() as temp_dir:
            self.app.directory = Path(temp_dir)

            self.app.save_1d_plots(response_name)

            for key in self.parameters.baseline.keys():
                self.assertTrue(Path(temp_dir, response_name, '1D', f'1D_{key}_{response_name}.png').exists())


    def test_save_1d_plots_exception(self):
        "test that the save 1d plots raises an exception if the directory is not set"
        response_name = 'total'

        #set poolsize to 1 due to issues with tests, generators, multiprocessing, and saving
        self.app.parallel.poolsize = 1

        self.app.run()

        with self.assertRaises(ValueError):
            self.app.save_1d_plots(response_name)


    def test_save_all_1d_plots(self):
        "test that the save all 1d plots works"

        #set poolsize to 1 due to issues with tests, generators, multiprocessing, and saving
        self.app.parallel.poolsize = 1

        self.app.run()

        with tempfile.TemporaryDirectory() as temp_dir:
            self.app.directory = Path(temp_dir)

            self.app.save_all_1d_plots()

            for response in self.app.response_variables():
                for key in self.parameters.baseline.keys():
                    self.assertTrue(Path(temp_dir, response, '1D', f'1D_{key}_{response}.png').exists())


    def test_get_2d_bounds(self):
        "test that the 2d bounds are correct"
        lb_x, ub_x, lb_y, ub_y = self.app.get_2d_bounds('x1___x2')

        self.assertEqual(lb_x, self.parameters.lower_bound['x1'])
        self.assertEqual(ub_x, self.parameters.upper_bound['x1'])
        self.assertEqual(lb_y, self.parameters.lower_bound['x2'])
        self.assertEqual(ub_y, self.parameters.upper_bound['x2'])


    def test_get_scale_linear(self):
        "test that the scale is correct"
        name, scale = ResponseSurface.get_scale(pd.Series([1,10], index=['x1', 'x2']))

        self.assertEqual(name, 'linear')
        self.assertIsInstance(scale, matplotlib.colors.Normalize)


    def test_get_scale_log(self):
        "test that the scale is correct"
        name, scale = ResponseSurface.get_scale(pd.Series([1,1000], index=['x1', 'x2']))

        self.assertEqual(name, 'log')
        self.assertIsInstance(scale, matplotlib.colors.LogNorm)


    def test_get_2d_plots(self):
        response_name = 'total'

        self.app.run()

        figures = list(self.app.get_2d_plots(response_name))

        for key, fig in figures:
            lb_x, ub_x, lb_y, ub_y = self.app.get_2d_bounds(key=key)

            x_name, y_name = key.split('___')

            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(fig.axes[0].get_xlabel(), x_name)
            self.assertEqual(fig.axes[0].get_ylabel(), y_name)
            self.assertEqual(fig.axes[0].get_title(), f'{x_name} vs {y_name} vs {response_name} linear scaling')
            self.assertEqual(fig.axes[0].get_xlim(), (lb_x, ub_x))
            self.assertEqual(fig.axes[0].get_ylim(), (lb_y, ub_y))
            plt.close(fig)


    def test_show_2d_plots(self):
        response_name = 'total'

        figures = [
            ('key1_key2', MagicMock()),
            ('key3_key4', MagicMock())
        ]

        self.app.get_2d_plots = MagicMock(return_value=figures)

        with unittest.mock.patch('matplotlib.pyplot.show'):
            self.app.show_2d_plots(response_name)

        self.assertEqual(self.app.get_2d_plots.call_count, 1)


    def test_generate_2d_plot_show_points_True(self):
        key = 'x1___x2'
        data = {'x1':    np.array([0, 0,   0, 0.5, 0.5, 0.5, 1, 1,   1]),
                'x2':    np.array([0, 0.5, 1, 0,   0.5, 1,   0, 0.5, 1]),
                'total': np.array([0, 0.5, 0, 0.5, 1,   0.5, 0, 0.5, 0])}
        response_name = 'total'
        lb_x = 0
        ub_x = 1
        lb_y = 0
        ub_y = 1
        show_points = True

        expected_title = f'x1 vs x2 vs {response_name} linear scaling'
        expected_xlabel = 'x1'
        expected_ylabel = 'x2'
        expected_xlim = (lb_x, ub_x)
        expected_ylim = (lb_y, ub_y)

        # mock the plt.subplots method to return a known figure object
        fig = plt.Figure()

        # call the method being tested
        result = ResponseSurface._generate_2d_plot((key, data, response_name, lb_x, ub_x, lb_y, ub_y, show_points))

        # assert that the expected figure object was returned
        self.assertEqual(result[0], key)
        self.assertIsInstance(result[1], plt.Figure)

        # assert that the figure object has the expected properties
        ax = result[1].axes[0]
        self.assertEqual(ax.get_title(), expected_title)
        self.assertEqual(ax.get_xlabel(), expected_xlabel)
        self.assertEqual(ax.get_ylabel(), expected_ylabel)
        self.assertEqual(ax.get_xlim(), expected_xlim)
        self.assertEqual(ax.get_ylim(), expected_ylim)

        # assert that the expected plot elements were added to the figure
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)
        self.assertEqual(len(ax.patches), 0)
        self.assertEqual(len(ax.collections[0].get_offsets()), len(data[expected_xlabel]))
        plt.close(fig)


    def test_generate_2d_plot_show_points_False(self):
        key = 'x1___x2'
        data = {'x1':    np.array([0, 0,   0, 0.5, 0.5, 0.5, 1, 1,   1]),
                'x2':    np.array([0, 0.5, 1, 0,   0.5, 1,   0, 0.5, 1]),
                'total': np.array([0, 0.5, 0, 0.5, 1,   0.5, 0, 0.5, 0])}
        response_name = 'total'
        lb_x = 0
        ub_x = 1
        lb_y = 0
        ub_y = 1
        show_points = False

        expected_title = f'x1 vs x2 vs {response_name} linear scaling'
        expected_xlabel = 'x1'
        expected_ylabel = 'x2'
        expected_xlim = (lb_x, ub_x)
        expected_ylim = (lb_y, ub_y)

        # mock the plt.subplots method to return a known figure object
        fig = plt.Figure()

        # call the method being tested
        result = ResponseSurface._generate_2d_plot((key, data, response_name, lb_x, ub_x, lb_y, ub_y, show_points))

        # assert that the expected figure object was returned
        self.assertEqual(result[0], key)
        self.assertIsInstance(result[1], plt.Figure)

        # assert that the figure object has the expected properties
        ax = result[1].axes[0]
        self.assertEqual(ax.get_title(), expected_title)
        self.assertEqual(ax.get_xlabel(), expected_xlabel)
        self.assertEqual(ax.get_ylabel(), expected_ylabel)
        self.assertEqual(ax.get_xlim(), expected_xlim)
        self.assertEqual(ax.get_ylim(), expected_ylim)

        # assert that the expected plot elements were added to the figure
        self.assertEqual(len(ax.collections), 0)
        self.assertEqual(len(ax.lines), 0)
        self.assertEqual(len(ax.patches), 0)
        plt.close(fig)


    def test_save_2d_plots(self):
        "test that the save 2d plots works"

        #set poolsize to 1 due to issues with tests, generators, multiprocessing, and saving
        self.app.parallel.poolsize = 1

        self.app.run()

        response_name = 'total'

        with tempfile.TemporaryDirectory() as temp_dir:
            self.app.directory = Path(temp_dir)

            self.app.save_2d_plots(response_name)

            for key in self.app.solver.get_response_2d():
                self.assertTrue(Path(temp_dir, response_name, '2D', f'2D_{key}_{response_name}.png').exists())


    def test_save_2d_plots_exception(self):
        "test that the save 2d plots raises an exception if the directory is not set"

        #set poolsize to 1 due to issues with tests, generators, multiprocessing, and saving
        self.app.parallel.poolsize = 1

        response_name = 'total'

        self.app.run()

        with self.assertRaises(ValueError):
            self.app.save_2d_plots(response_name)



    def test_save_all_2d_plots(self):
        "test that the save all 2d plots works"

        #set poolsize to 1 due to issues with tests, generators, multiprocessing, and saving
        self.app.parallel.poolsize = 1

        self.app.run()

        with tempfile.TemporaryDirectory() as temp_dir:
            self.app.directory = Path(temp_dir)

            self.app.save_all_2d_plots()

            keys = list(self.app.solver.get_response_2d().keys())

            for response in self.app.response_variables():
                for key in keys:
                    self.assertTrue(Path(temp_dir, response, '2D', f'2D_{key}_{response}.png').exists())


    def test_save_all_plots(self):
        self.app.save_all_1d_plots = MagicMock()
        self.app.save_all_2d_plots = MagicMock()
        self.app.save_all_tornado_plots = MagicMock()

        self.app.save_all_plots()

        self.app.save_all_1d_plots.assert_called_once()
        self.app.save_all_2d_plots.assert_called_once()
        self.app.save_all_tornado_plots.assert_called_once()


    def test_save_all_response_plot(self):
        self.app.save_1d_plots = MagicMock()
        self.app.save_2d_plots = MagicMock()
        self.app.save_tornado_plot = MagicMock()

        response_name = 'response'

        self.app.save_all_response_plot(response_name)

        self.app.save_1d_plots.assert_called_once_with(response_name=response_name)
        self.app.save_2d_plots.assert_called_once_with(response_name=response_name)
        self.app.save_tornado_plot.assert_called_once_with(response_name=response_name)


    def test_get_data_bounds(self):
        "test that the data bounds are correct"
        self.app.run()

        data_1d = self.app.solver.get_response_1d()

        response_name = "total"

        lb, ub = self.app.get_data_bounds(data_1d, response_name)

        self.assertEqual(lb, -2.0)
        self.assertEqual(ub, 2.0)


    def test_rescale_data_percent(self):
        "test that the data is rescaled correctly"
        self.app.parameters.lower_bound = pd.Series({'x1': 0, 'x2': 0, 'x3': 0})
        self.app.parameters.upper_bound = pd.Series({'x1': 2, 'x2': 2, 'x3': 2})
        self.app.parameters.baseline = pd.Series({'x1': 1, 'x2': 1, 'x3': 1})

        response_name = "total"

        data_1d = {
            'x1': pd.DataFrame({"total":[5, 10, 15], "x1":[0, 1, 2]}),
            'x2': pd.DataFrame({"total":[9, 10, 11], "x2":[0, 1, 2]}),
            'x3': pd.DataFrame({"total":[1, 10, 19], "x3":[0, 1, 2]})
        }

        self.app.solver.get_response_1d = MagicMock(return_value=data_1d)
        self.app.solver.get_baseline_response = MagicMock(return_value=pd.DataFrame({'total': [10]}))

        data, rescale_name = self.app.rescale_data(response_name)

        self.assertEqual(rescale_name, f"Percent Change from 1.00e+01 for {response_name}")
        pd.testing.assert_frame_equal(data['x1'], pd.DataFrame({"x1":[0, 0.5, 1.0], "total":[0.5, 1.0, 1.5]}))
        pd.testing.assert_frame_equal(data['x2'], pd.DataFrame({"x2":[0, 0.5, 1.0], "total":[0.9, 1.0, 1.1]}))
        pd.testing.assert_frame_equal(data['x3'], pd.DataFrame({"x3":[0, 0.5, 1.0], "total":[0.1, 1.0, 1.9]}))



    def test_rescale_data_center(self):
        "test that the data is rescaled correctly"
        self.app.parameters.lower_bound = pd.Series({'x1': 0, 'x2': 0, 'x3': 0})
        self.app.parameters.upper_bound = pd.Series({'x1': 2, 'x2': 2, 'x3': 2})
        self.app.parameters.baseline = pd.Series({'x1': 1, 'x2': 1, 'x3': 1})

        response_name = "total"

        data_1d = {
            'x1': pd.DataFrame({"total":[-10, 0, 10], "x1":[0, 1, 2]}),
            'x2': pd.DataFrame({"total":[-5, 0, 5], "x2":[0, 1, 2]}),
            'x3': pd.DataFrame({"total":[-1, 0, 1], "x3":[0, 1, 2]})
        }

        self.app.solver.get_response_1d = MagicMock(return_value=data_1d)
        self.app.solver.get_baseline_response = MagicMock(return_value=pd.DataFrame({'total': [0]}))

        data, rescale_name = self.app.rescale_data(response_name)

        self.assertEqual(rescale_name, f"Absolute Change from 0.00e+00 for {response_name} with Linear scaling")
        pd.testing.assert_frame_equal(data['x1'], pd.DataFrame({"x1":[0, 0.5, 1.0], "total":[-1.0, 0.0, 1.0]}))
        pd.testing.assert_frame_equal(data['x2'], pd.DataFrame({"x2":[0, 0.5, 1.0], "total":[-0.5, 0, 0.5]}))
        pd.testing.assert_frame_equal(data['x3'], pd.DataFrame({"x3":[0, 0.5, 1.0], "total":[-0.1, 0, 0.1]}))


    def test_gradient_bar(self):
        ax = MagicMock()
        data = pd.DataFrame({'x': [1, 2, 3], 'color': [0.1, 0.5, 0.9]})
        x_column = 'x'
        color_column = 'color'
        vertical_position = 0.5
        height = 0.5

        self.app.gradient_bar(ax, data, x_column, color_column, vertical_position, height)

        ax.scatter.assert_called_once()


    def test_get_bar_order(self):
        "test that the bar order is correct"
        data_1d = {
            'key1': pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'response': [0.1, 0.2, 0.3]}),
            'key2': pd.DataFrame({'x': [4, 5, 6], 'y': [7, 8, 9], 'response': [0.4, 0.5, 1.6]}),
            'key3': pd.DataFrame({'x': [7, 8, 9], 'y': [10, 11, 12], 'response': [0.7, 0.8, 1.3]})
        }
        response_name = 'response'

        expected_order = ['key1', 'key3', 'key2']
        actual_order = self.app.get_bar_order(data_1d, response_name)

        self.assertEqual(actual_order, expected_order)


    def test_get_tornado_plot(self):
        self.app.parameters.lower_bound = pd.Series({'x1': 0, 'x2': 0, 'x3': 0})
        self.app.parameters.upper_bound = pd.Series({'x1': 2, 'x2': 2, 'x3': 2})
        self.app.parameters.baseline = pd.Series({'x1': 1, 'x2': 1, 'x3': 1})

        data_1d = {
            'x1': pd.DataFrame({"total":[-10, 0, 10], "x1":[0, 1, 2]}),
            'x2': pd.DataFrame({"total":[-5, 0, 5], "x2":[0, 1, 2]}),
            'x3': pd.DataFrame({"total":[-1, 0, 1], "x3":[0, 1, 2]})
        }

        self.app.solver.get_response_1d = MagicMock(return_value=data_1d)
        self.app.solver.get_baseline_response = MagicMock(return_value=pd.DataFrame({'total': [0]}))


        fig = self.app.get_tornado_plot('total')

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 2)
        self.assertIsInstance(fig.axes[0], plt.Axes)
        self.assertIsInstance(fig.axes[1], plt.Axes)
        self.assertEqual(fig.axes[0].get_ylabel(), 'Parameter')
        self.assertEqual(len(fig.axes[1].collections), 2)
        plt.close(fig)


    def test_show_tornado_plot(self):
        response_name = 'total'

        fig = MagicMock()

        self.app.get_tornado_plot = MagicMock(return_value=fig)

        with unittest.mock.patch('matplotlib.pyplot.show'):
            self.app.show_tornado_plot(response_name)

        self.assertEqual(self.app.get_tornado_plot.call_count, 1)


    def test_save_tornado_plot(self):
        "test that the save tornado plot works"
        self.app.run()

        response_name = 'total'

        with tempfile.TemporaryDirectory() as temp_dir:
            self.app.directory = Path(temp_dir)

            self.app.save_tornado_plot(response_name)

            self.assertTrue(Path(temp_dir, 'tornado', f'tornado_{response_name}.png').exists())


    def test_save_tornado_plot_exception(self):
        "test that the save tornado plot raises an exception if the directory is not set"
        response_name = 'total'

        self.app.run()

        with self.assertRaises(ValueError):
            self.app.save_tornado_plot(response_name)


    def test_save_all_tornado_plots(self):
        "test that the save all tornado plots works"
        self.app.run()

        with tempfile.TemporaryDirectory() as temp_dir:
            self.app.directory = Path(temp_dir)

            self.app.save_all_tornado_plots()

            for response in self.app.response_variables():
                self.assertTrue(Path(temp_dir, 'tornado', f'tornado_{response}.png').exists())


    def test_save(self):
        "test that the save method works"
        self.app.run()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            self.app.directory = temp_dir
            self.app.save()
            self.assertTrue((temp_dir / "response_surface.h5").exists())


    def test_save_no_directory(self):
        "test saving without a directory raises an exception"
        with self.assertRaises(ValueError):
            self.app.save()


    def test_save_not_run(self):
        "test saving without running raises an exception"
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                self.app.directory = temp_dir
                self.app.save()


    def test_load(self):
        "test that the load method works"
        self.app.run()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            self.app.directory = temp_dir
            self.app.save()

            app = ResponseSurface.load(directory=temp_dir, parallel=Parallel(ordered=False))

            pd.testing.assert_frame_equal(app.solver.response, self.app.solver.response)
            pd.testing.assert_frame_equal(app.solver.request, self.app.solver.request)


class TestResponseSurfaceExceptionNan(unittest.TestCase):

    def test_run_nan(self):
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_2':4, 'x_3':5, 'nan_lower_bound' : 0.5, 'nan_upper_bound':1, 'n':4,
                   'exception_lower_bound':-1, 'exception_upper_bound':-0.5} #variables we want to keep fixed
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernelNanException(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        parameters = ResponseSamplerParameters(
                        lower_bound=pd.Series({'x_1':1, 'nan_trigger':0, 'exception_trigger':0}),
                        upper_bound=pd.Series({'x_1':5, 'nan_trigger':1, 'exception_trigger':1}),
                        baseline=pd.Series({'x_1':3, 'nan_trigger':0, 'exception_trigger':0}),
                        points_1d=5,
                        points_2d_per_dimension=0)

        app = ResponseSurface(parameters=parameters,
                             batch=batch,
                             parallel=parallel,
                             directory=None,
                             processing_function=get_total)




        app.run()

        self.assertIsInstance(app.solver.response, pd.DataFrame)
        self.assertIsInstance(app.solver.request, pd.DataFrame)
        self.assertTupleEqual(app.solver.response.shape, (10, 1))
        self.assertTupleEqual(app.solver.request.shape, (10, 3))

    def test_run_nan_exception(self):
        "test a mixture of nan and exceptions is filtered"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_2':4, 'x_3':5, 'nan_lower_bound' : 0.5, 'nan_upper_bound':1, 'n':4,
                   'exception_lower_bound':0.5, 'exception_upper_bound':1.0} #variables we want to keep fixed
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernelNanException(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        parameters = ResponseSamplerParameters(
                        lower_bound=pd.Series({'x_1':1, 'nan_trigger':0, 'exception_trigger':0}),
                        upper_bound=pd.Series({'x_1':5, 'nan_trigger':1, 'exception_trigger':1}),
                        baseline=pd.Series({'x_1':3, 'nan_trigger':0, 'exception_trigger':0}),
                        points_1d=5,
                        points_2d_per_dimension=0)

        app = ResponseSurface(parameters=parameters,
                             batch=batch,
                             parallel=parallel,
                             directory=None,
                             processing_function=get_total)




        app.run()

        self.assertIsInstance(app.solver.response, pd.DataFrame)
        self.assertIsInstance(app.solver.request, pd.DataFrame)
        self.assertTupleEqual(app.solver.response.shape, (7, 1))
        self.assertTupleEqual(app.solver.request.shape, (7, 3))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResponseSurface)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestResponseSurfaceExceptionNan)
    unittest.TextTestRunner(verbosity=2).run(suite)
