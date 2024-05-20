# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the SALib sensitivity analysis"

import unittest

import attrs
import matplotlib.axes
import numpy as np
import pandas as pd
import scipy.stats

from gumps.solvers.sensitivity import (SensitivitySolver,
                                       SensitivitySolverParameters)


class TestSensitivitySolverParameters(unittest.TestCase):
    "test the sensitivity solver parameters"

    def setUp(self) -> None:
        "set up the test"
        self.params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))



    def test_initialize(self):
        "test the initialization of the sensitivity solver parameters"
        params = SensitivitySolverParameters(**attrs.asdict(self.params))

        self.assertEqual(params.sample_power, 5)
        self.assertEqual(params.calc_second_order, True)
        self.assertEqual(params.seed, None)

    def test_total_points(self):
        "test the total_points property"
        self.params.sample_power = 2
        self.assertEqual(self.params.total_points, 24)

    def test_total_points_second_order_false(self):
        "test the total_points property"
        self.params.sample_power = 2
        self.params.calc_second_order = False
        self.assertEqual(self.params.total_points, 16)

    def test_N(self):
        "test the N property"
        self.params.sample_power = 2
        self.assertEqual(self.params.N, 4)

    def test_D(self):
        "test the D property"
        self.assertEqual(self.params.D, 2)

    def test_get_problem(self):
        "test the get_problem method"
        problem = self.params.get_problem()
        self.assertEqual(problem["num_vars"], 2)
        self.assertEqual(problem["names"], ["x1", "x2"])
        self.assertEqual(problem["bounds"], [[0, 1], [0, 1]])


class TestSensitivity(unittest.TestCase):
    "test SALib sensitivity analysis"

    def test_init(self):
        "test the initialization of the PCA solver"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        self.assertIsInstance(solver.solver_settings, SensitivitySolverParameters)
        self.assertTrue(solver._has_next)
        self.assertIsNone(solver.analysis)

    def test_has_next_true(self):
        "test the has_next method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        self.assertTrue(solver.has_next())

    def test_has_next_false(self):
        "test the has_next method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        solver._has_next = False
        self.assertFalse(solver.has_next())

    def test_ask(self):
        "test the ask method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        self.assertIsInstance(solver.ask(), np.ndarray)
        self.assertTrue(np.all(solver.ask() >= 0))
        self.assertTrue(np.all(solver.ask() <= 1))
        self.assertEqual(solver.ask().shape, (192, 2))

    def test_tell(self):
        "test the tell method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        ask = solver.ask()
        loss = np.ones(ask.shape[0])
        solver.tell(np.squeeze(loss))

    def test_tell_exception_sample_length(self):
        "test the tell method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        points = scipy.stats.qmc.Sobol(1).random(params.total_points)

        with self.assertRaises(RuntimeError):
            solver.tell(np.squeeze(points))

    def test_tell_exception_wrong_sample_length(self):
        "test the tell method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        ask = solver.ask()
        loss = np.ones(ask.shape[0]-2)
        with self.assertRaises(ValueError):
            solver.tell(np.squeeze(loss))

    def test_plot_exception(self):
        "test the plot method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        with self.assertRaises(RuntimeError):
            solver.plot()

    def test_plot(self):
        "test the plot method"
        params = SensitivitySolverParameters(lower_bound=pd.Series({'x1': 0, 'x2': 0}),
                                                    upper_bound=pd.Series({'x1': 1, 'x2': 1}))
        solver = SensitivitySolver(solver_settings=params)
        ask = solver.ask()
        loss = np.ones(ask.shape[0])
        solver.tell(np.squeeze(loss))
        for ax in solver.plot():
            self.assertIsInstance(ax, matplotlib.axes.Axes)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSensitivitySolverParameters)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSensitivity)
    unittest.TextTestRunner(verbosity=2).run(suite)
