# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np
import scipy.stats
import xarray as xr
import pandas as pd

import gumps.solvers.monte_carlo_timeseries_solver


def get_dataset(runs:int, components:int, times:int) -> xr.Dataset:
    "get a randomly generated data set, this just ensure we get something of the right shape"
    data = np.random.rand( runs, components, times)

    dataset = xr.Dataset(
        {
            'data': (['index', 'component', 'time'], data)
        },
        coords={
            'index': np.arange(runs),
            'component': np.arange(components),
            'time': np.arange(times)
        }
    )
    return dataset

def get_quantiles(dataset:xr.Dataset, probability:float) -> xr.Dataset:
    "get the quantiles from a dataset"
    quantiles = dataset.quantile(probability, dim='index')

    return quantiles

class TestMonteCarloTimeSeries(unittest.TestCase):
    "test the monte carlo solver"

    def setUp(self) -> None:
        "setup the test case"
        self.distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        self.parameters = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloParameters(variable_distributions=self.distributions,
                                                                                           target_probability=0.5)

        self.solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)


    def test_create(self):
        "test creation of MonteCarloSolver"
        self.assertEqual(self.solver.solver_settings, self.parameters)

    def test_batch_size(self):
        "test batch size setup"
        self.assertEqual(self.solver.batch_size, 64)

    def test_min_batch_size(self):
        "test batch size setup"
        self.parameters.runnable_batch_size=128

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        self.assertEqual(solver.batch_size, 128)

    def test_tell_none(self):
        "test the tell interface"
        np.random.seed(0)

        quantiles = get_quantiles(get_dataset(1000, 4, 10), self.parameters.target_probability)
        quantiles = xr.concat([quantiles, ], dim="iteration")

        self.solver.tell(loss=quantiles)

        xr.testing.assert_equal(self.solver.history, quantiles)

    def test_tell(self):
        "test the tell interface"
        np.random.seed(0)

        quantiles = get_quantiles(get_dataset(1000, 4, 10), self.parameters.target_probability)
        quantiles2 = get_quantiles(get_dataset(1000, 4, 10), self.parameters.target_probability)

        correct = xr.concat([quantiles, quantiles2], dim='iteration')

        self.solver.tell(loss=quantiles)
        self.solver.tell(loss=quantiles2)

        xr.testing.assert_equal(self.solver.history, correct)

    def test_next_none(self):
        "test if the has_next is True when history is None"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=parameters)

        self.assertTrue(solver.has_next())

    def test_next_one(self):
        "test if the has_next is True when history has one entry"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=parameters)

        self.assertTrue(solver.has_next())

    def test_next_true_short(self):
        "test has has_next is true on a sequence shorter than window but greater than min steps"
        self.parameters.tolerance = 1e-1
        self.parameters.min_steps = 2

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        np.random.seed(0)
        quantiles = get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability)
        quantiles2 = get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability)
        quantiles3 = get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability)

        solver.tell(loss=quantiles)
        solver.tell(loss=quantiles2)
        solver.tell(loss=quantiles3)

        self.assertTrue(solver.has_next())

    def test_next_false_short(self):
        "test has has_next is false on a sequence shorter than window but greater than min steps"
        self.parameters.tolerance = 1e-1
        self.parameters.min_steps = 2

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        np.random.seed(0)
        quantiles = get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability)

        solver.tell(loss=quantiles)
        solver.tell(loss=quantiles)
        solver.tell(loss=quantiles)

        self.assertFalse(solver.has_next())

    def test_get_convergence(self):
        "test get convergence"
        self.parameters.tolerance = 1e-1
        self.parameters.min_steps = 2
        self.parameters.window = 4

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        np.random.seed(0)
        solver.tell(loss=get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability))

        convergence = solver.get_convergence()

        self.assertEqual(convergence.shape, (1, 4, 10))
        self.assertTrue(convergence.all())


    def test_get_convergence_exception(self):
        "test that an exception is raised if convergence is called before history is available"
        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        with self.assertRaises(RuntimeError):
            solver.get_convergence()


    def test_next_true_window(self):
        "test has has_next is true on a sequence larger than window"
        self.parameters.tolerance = 1e-1
        self.parameters.min_steps = 2
        self.parameters.window = 4

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        np.random.seed(0)
        solver.tell(loss=get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(10, 4, 10), self.parameters.target_probability))

        self.assertTrue(solver.has_next())

    def test_next_false_window(self):
        "test has has_next is false on a sequence larger than window"
        self.parameters.tolerance = 1e-1
        self.parameters.min_steps = 2
        self.parameters.window = 4

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        np.random.seed(0)
        solver.tell(loss=get_quantiles(get_dataset(100, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(100, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(100, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(100, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(100, 4, 10), self.parameters.target_probability))
        solver.tell(loss=get_quantiles(get_dataset(100, 4, 10), self.parameters.target_probability))

        self.assertFalse(solver.has_next())

    def test_ask(self):
        "test the ask interface returns the appropriate items"
        distributions = {'a':scipy.stats.uniform(0.0, 1), 'b':scipy.stats.norm(0.0, 1)}

        self.parameters.tolerance = 1e-1
        self.parameters.min_steps = 2
        self.parameters.window = 4
        self.parameters.sampler_seed = 0
        self.parameters.sampler_scramble = False
        self.parameters.variable_distributions = distributions

        solver = gumps.solvers.monte_carlo_timeseries_solver.MonteCarloTimeSeriesSolver(solver_settings=self.parameters)

        params = solver.ask()

        correct_first = pd.DataFrame({'a':[0.005, 0.5, 0.7475],
                                      'b':[-2.576, 0.0, -0.666]},
                                        index=[0, 1, 2])

        correct_last = pd.DataFrame({'a':[0.76296875, 0.51546875, 0.02046875],
                                     'b':[0.1165868, -0.52754885, 0.82005018]},
                                        index=[61, 62, 63])

        pd.testing.assert_frame_equal(params.iloc[:3,:], correct_first, rtol=1e-1)
        pd.testing.assert_frame_equal(params.iloc[-3:,:], correct_last, rtol=1e-1)
        self.assertEqual(params.shape, (64,2))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMonteCarloTimeSeries)
    unittest.TextTestRunner(verbosity=2).run(suite)
