# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the monte carlo application"

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats

import gumps.apps.monte_carlo_timeseries
import gumps.solvers.monte_carlo_timeseries_solver
import gumps.studies.batch_time_study_example
import gumps.studies.batch_time_study

import gumps.common.parallel

def get_dataset(target_probability, times, distributions) -> xr.Dataset:
    columns = [f't{i}' for i in distributions.keys()]
    correct_np = np.array([dist.ppf(target_probability) for dist in distributions.values()]).T

    #create 2d numpy arrays for each target_probability
    seq = []
    for row in correct_np:
        temp = row[:, np.newaxis] * times[np.newaxis, :] + row[:, np.newaxis]
        seq.append(temp)

    #convert to dataframes
    dfs = []
    for row in seq:
        df = pd.DataFrame(row.T, columns=columns)
        df['time'] = times
        df = df.set_index('time')
        dfs.append(df)

    array = xr.concat([df.to_xarray() for df in dfs], dim='quantile')

    array = array.assign_coords(quantile=target_probability)

    return array

def exception_handler(function, x, e):
    return None

def processing_function(frame: pd.DataFrame|None) -> pd.DataFrame|None:
    "process the dataframe for the loss function"
    if frame is None:
        return None
    return frame

class TestMonteCarloApp(unittest.TestCase):
    "test the monte carlo app"

    def setUp(self):
        "create a basic setup for all the tests"

        #make sure this test is repeatable
        np.random.seed(0)

        self.distributions = {'a':scipy.stats.uniform(0.9, 1.1),
                              'b':scipy.stats.norm(1, 1e-1),
                              'c':scipy.stats.uniform(0.5, 1.5),
                              'd':scipy.stats.norm(1, 1e-2)}

        self.parameters = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloParameters(variable_distributions=self.distributions,
                                                                                        target_probability=[0.4, 0.5, 0.6],
                                                                                        window=2,
                                                                                        tolerance=1e-1,
                                                                                        min_steps=2,
                                                                                        sampler_seed=0,
                                                                                        sampler_scramble=False)

        self.batch = gumps.studies.batch_time_study_example.BatchTimeSeriesExample()

    def test_monte_carlo(self):
        "integration test for monte carlo solver"
        correct = get_dataset(self.parameters.target_probability, self.batch.time_series, self.distributions)

        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)
        app.run()

        answer = app.answer()

        np.testing.assert_allclose(answer.to_array(), correct.to_array(),
                                   rtol=1e-1,
                                   atol=1e-1)


    def test_answer_exception(self):
        "test that calling answer before running raises an exception"
        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)

        with self.assertRaises(RuntimeError):
            app.answer()


    def test_save_data_hdf5_directory_exception(self):
        "test that calling save_data_hdf5 before running raises an exception"
        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)

        with self.assertRaises(RuntimeError):
            app.save_data_hdf5()

    def test_save_data_hdf5_data_exception(self):
        "test that calling save_data_hdf5 before running raises an exception"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
                processing_function=None,
                directory=directory,
                batch=self.batch)

            with self.assertRaises(RuntimeError):
                app.save_data_hdf5()

    def test_load_data_hdf5_directory_exception(self):
        "test that calling load_data_hdf5 before running raises an exception"
        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)

        with self.assertRaises(RuntimeError):
            app.load_data_hdf5()

    def test_get_plots_exception(self):
        "test that get plots raises an exception if there is no data to plot"
        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)

        with self.assertRaises(RuntimeError):
            app.get_plots()

    def test_monte_carlo_nan(self):
        "integration test for monte carlo solver for proper handling of nans in the batch study"

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"

            #1% chance to set a few elements to nan
            random_fraction = 0.01

            if np.random.random() < random_fraction:
                frame.iloc[0, 0] = np.nan
                frame.iloc[1, 1] = np.nan
                frame.iloc[2, 2] = np.nan
                frame.iloc[3, 3] = np.nan
            return frame

        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=processing_function,
            directory=None,
            batch=self.batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(len(cm.output) > 0)

        answer = app.answer()

        correct = get_dataset(self.parameters.target_probability, self.batch.time_series, self.distributions)

        np.testing.assert_allclose(answer.to_array(), correct.to_array(),
                                   rtol=1e-1,
                                   atol=1e-1)


    def test_monte_carlo_nan_exception(self):
        "integration test for monte carlo solver for proper handling of nans in the batch study"

        distributions = {'a':scipy.stats.uniform(0.9, 1.1),
                              'b':scipy.stats.norm(1, 1e-1),
                              'c':scipy.stats.uniform(0.5, 1.5),
                              'd':scipy.stats.norm(1, 1e-2),
                              'nan':scipy.stats.uniform(0, 1),
                              'fail':scipy.stats.uniform(0, 1)}

        parameters = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloParameters(variable_distributions=distributions,
                                                                                        target_probability=[0.4, 0.5, 0.6],
                                                                                        window=2,
                                                                                        tolerance=1e-1,
                                                                                        min_steps=2,
                                                                                        sampler_seed=0,
                                                                                        sampler_scramble=False)

        study = gumps.studies.study.TestStudyTime(model_variables={'time_start':0, 'time_end':10, 'time_points':100},
                                                  problem ={ 'nan': 0.0, 'fail': 0.0,
                                                            'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0})

        parallel = gumps.common.parallel.Parallel(poolsize=4)

        batch = gumps.studies.batch_time_study.BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=parameters,
            processing_function=processing_function,
            directory=None,
            batch=batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(len(cm.output) > 0)

        answer = app.answer()

        del distributions['nan']
        del distributions['fail']

        correct = get_dataset(parameters.target_probability, study.time_series, distributions)

        np.testing.assert_allclose(answer.to_array(), correct.to_array(),
                                   rtol=1e-1,
                                   atol=1e-1)

    def test_monte_carlo_display_results(self):
        "integration test for monte carlo solver"
        self.parameters.min_steps = 3
        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)

        with self.assertLogs('gumps.apps.monte_carlo_timeseries', level='INFO') as cm:
            app.run()

        self.assertTrue(len(cm.output) > 0)

        correct = ['INFO:gumps.apps.monte_carlo_timeseries:Gen= 0 Convergence= 1200/1200 = 100.0%',
                   'INFO:gumps.apps.monte_carlo_timeseries:Gen= 1 Convergence= 1200/1200 = 100.0%',
                   'INFO:gumps.apps.monte_carlo_timeseries:Gen= 2 Convergence= 1200/1200 = 100.0%',
                   'INFO:gumps.apps.monte_carlo_timeseries:Gen= 3 Convergence= 1200/1200 = 100.0%']

        self.assertEqual(cm.output, correct)


    def test_pre_processing_function(self):
        "integration test for monte carlo solver"
        def pre_processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame["a"])

        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            pre_processing_function=pre_processing_function,
            processing_function=None,
            directory=None,
            batch=self.batch)
        app.run()

        answer = app.answer()

        correct = get_dataset(self.parameters.target_probability, self.batch.time_series, {'a':self.distributions['a']})

        np.testing.assert_allclose(answer.to_array(), correct.to_array(),
                                   rtol=1e-1,
                                   atol=1e-1)


    def test_monte_carlo_single_prob(self):
        "integration test for monte carlo solver"
        self.parameters.target_probability = [0.4]

        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)
        app.run()

        answer = app.answer()

        correct = get_dataset(self.parameters.target_probability, self.batch.time_series, self.distributions)

        np.testing.assert_allclose(answer.to_array(), correct.to_array(),
                                   rtol=1e-1,
                                   atol=1e-1)


    def test_monte_carlo_scalar(self):
        "integration test for monte carlo solver"
        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame["ta"])

        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=processing_function,
            directory=None,
            batch=self.batch)
        app.run()

        answer = app.answer()

        correct = get_dataset(self.parameters.target_probability, self.batch.time_series, {'a':self.distributions['a']})

        np.testing.assert_allclose(answer.to_array(), correct.to_array(),
                                   rtol=1e-1,
                                   atol=1e-1)


    def test_get_plots(self):
        "test that the get plots function works"
        app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
            processing_function=None,
            directory=None,
            batch=self.batch)

        app.run()

        plots = app.get_plots()

        self.assertListEqual(list(plots.keys()), ['ta', 'tb', 'tc', 'td', 'corner'])

    def test_monte_carlo_plotting_scalar(self):
        "test that plotting works for the monte carlo solver"
        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame["ta"])

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
                processing_function=processing_function,
                directory=directory,
                batch=self.batch)
            app.run()

            app.create_plots()

            self.assertEqual(len(list(Path(directory).glob("*.png"))), 2)

    def test_monte_carlo_plotting_multi(self):
        "test that plotting works for the monte carlo solver"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
                processing_function=None,
                directory=directory,
                batch=self.batch)
            app.run()

            app.create_plots()

            self.assertEqual(len(list(Path(directory).glob("*.png"))), 5)


    def test_monte_carlo_plotting_multi_no_directory(self):
        "test that plotting works for the monte carlo solver"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
                processing_function=None,
                directory=None,
                batch=self.batch)
            app.run()

            app.create_plots()


    def test_monte_carlo_save_load(self):
        "test that saving works for the monte carlo solver"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
                processing_function=None,
                directory=directory,
                batch=self.batch)
            app.run()

            app.save_data_hdf5()

            app2 = gumps.apps.monte_carlo_timeseries.MonteCarloTimeSeriesApp(parameters=self.parameters,
                processing_function=None,
                directory=directory,
                batch=self.batch)

            app2.load_data_hdf5()

            np.testing.assert_almost_equal(app2.scores.to_array().to_numpy(), app.scores.to_array().to_numpy())
            np.testing.assert_almost_equal(app2.chain.to_numpy(), app.chain.to_numpy())
            np.testing.assert_almost_equal(app2.parameters.target_probability, app.parameters.target_probability)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMonteCarloApp)
    unittest.TextTestRunner(verbosity=2).run(suite)
