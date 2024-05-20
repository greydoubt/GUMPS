# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import numpy as np
import pandas as pd
import scipy.stats

import gumps.solvers.monte_carlo_solver


class TestMonteCarlo(unittest.TestCase):
    "test the monte carlo solver"

    def test_create(self):
        "test creation of MonteCarloSolver"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        self.assertEqual(solver.solver_settings, parameters)

    def test_batch_size(self):
        "test batch size setup"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        self.assertEqual(solver.batch_size, 1024)

    def test_min_batch_size(self):
        "test batch size setup"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933,
            runnable_batch_size=2048)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        self.assertEqual(solver.batch_size, 2048)

    def test_tell_none(self):
        "test the tell interface"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        add = pd.Series({'d_0':1.0, 'd_1':2.0})

        solver.tell(loss=add)

        np.testing.assert_array_equal(solver.history, np.atleast_2d(add))

    def test_tell(self):
        "test the tell interface"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        add = pd.Series({'d_0':1.0, 'd_1':2.0})
        add2 = pd.Series({'d_0':3.0, 'd_1':4.0})

        correct = pd.DataFrame({'d_0':[1.0, 3.0], 'd_1':[2.0, 4.0]})

        solver.tell(loss=add)
        solver.tell(loss=add2)

        pd.testing.assert_frame_equal(solver.history, correct)

    def test_next_none(self):
        "test if the has_next is True when history is None"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        self.assertTrue(solver.has_next())

    def test_next_one(self):
        "test if the has_next is True when history has one entry"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        self.assertTrue(solver.has_next())

    def test_next_true_short(self):
        "test has has_next is true on a sequence shorter than window but greter than min steps"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933,
                tolerance=1e-1, min_steps=2)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        solver.tell(loss=pd.Series({'d_0':1.0, 'd_1':2.0}))
        solver.tell(loss=pd.Series({'d_0':2.0, 'd_1':3.0}))
        solver.tell(loss=pd.Series({'d_0':3.0, 'd_1':4.0}))

        self.assertTrue(solver.has_next())

    def test_next_false_short(self):
        "test has has_next is false on a sequence shorter than window but greter than min steps"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933,
                tolerance=1e-1, min_steps=2)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        solver.tell(loss=pd.Series({'d_0':1.0, 'd_1':2.0}))
        solver.tell(loss=pd.Series({'d_0':1.01, 'd_1':2.05}))
        solver.tell(loss=pd.Series({'d_0':1.09, 'd_1':2.01}))

        self.assertFalse(solver.has_next())

    def test_next_true_window(self):
        "test has has_next is true on a sequence larger than window"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933,
                window=4, tolerance=1e-1, min_steps=2)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        solver.tell(loss=pd.Series({'d_0':1.0, 'd_1':2.0}))
        solver.tell(loss=pd.Series({'d_0':2.0, 'd_1':3.0}))
        solver.tell(loss=pd.Series({'d_0':3.0, 'd_1':4.0}))
        solver.tell(loss=pd.Series({'d_0':1.0, 'd_1':2.0}))
        solver.tell(loss=pd.Series({'d_0':2.0, 'd_1':3.0}))
        solver.tell(loss=pd.Series({'d_0':3.0, 'd_1':4.0}))

        self.assertTrue(solver.has_next())

    def test_next_false_window(self):
        "test has has_next is false on a sequence larger than window"
        distributions = {'a':scipy.stats.uniform(0.5, 2), 'b':scipy.stats.norm(0.5, 1e-3)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933,
                window=4, tolerance=1e-1, min_steps=2)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        solver.tell(loss=pd.Series({'d_0':1.0, 'd_1':2.0}))
        solver.tell(loss=pd.Series({'d_0':2.0, 'd_1':3.0}))
        solver.tell(loss=pd.Series({'d_0':1.05, 'd_1':2.01}))
        solver.tell(loss=pd.Series({'d_0':1.0, 'd_1':2.0}))
        solver.tell(loss=pd.Series({'d_0':1.01, 'd_1':2.05}))
        solver.tell(loss=pd.Series({'d_0':1.09, 'd_1':2.01}))

        self.assertFalse(solver.has_next())

    def test_ask(self):
        "test the ask interface returns the appropriate items"
        distributions = {'a':scipy.stats.uniform(0.0, 1), 'b':scipy.stats.norm(0.0, 1.0)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=0.933,
                window=4, tolerance=1e-1, min_steps=2, sampler_seed=0, sampler_scramble=False)

        solver = gumps.solvers.monte_carlo_solver.MonteCarloSolver(solver_settings=parameters)

        params = solver.ask()

        correct_first = pd.DataFrame({'a':[6.70000000e-04, 5.00000000e-01, 7.49665000e-01],
            'b':[-3.20727250e+00, 0.00000000e+00, -6.73435925e-01]})

        correct_last = pd.DataFrame({'a':[7.50640254e-01, 5.00975254e-01, 1.64525391e-03],
            'b':[7.33386279e-03, -6.64263741e-01, 6.82665117e-01]},
            index=[1021, 1022, 1023])

        pd.testing.assert_frame_equal(params.iloc[:3,:], correct_first)
        pd.testing.assert_frame_equal(params.iloc[-3:,:], correct_last)
        self.assertEqual(params.shape, (1024,2))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMonteCarlo)
    unittest.TextTestRunner(verbosity=2).run(suite)
