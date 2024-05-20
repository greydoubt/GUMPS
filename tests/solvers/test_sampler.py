# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#Create a test for the Sampler solver

import unittest
import numpy as np
import pandas as pd

from gumps.solvers.sampler import SamplerSolver, SamplerSolverParameters


class TestSampler(unittest.TestCase):
    "test the Sampler interface"

    def test_has_next_true(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 2,
            lower_bound = {'a':-10, 'b':-10},
            upper_bound = {'a':10, 'b':10},
            sampler = "sobol"
            )
        solver = SamplerSolver(solver_settings=solver_settings)
        self.assertEqual(solver.has_next(), True)

    def test_has_next_false(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 2,
            lower_bound = {'a':-10, 'b':-10},
            upper_bound = {'a':10, 'b':10},
            sampler = "sobol"
            )
        solver = SamplerSolver(solver_settings=solver_settings)
        solver.tell(pd.DataFrame([1,2]))
        self.assertEqual(solver.has_next(), False)

    def test_sample_variable(self):
        with self.assertRaises(ValueError) as cm:
            SamplerSolverParameters(
            number_of_samples = 2,
            lower_bound = {'a':-10, 'b':-10},
            upper_bound = {'a':10, 'b':10},
            sampler = "None"
            )
        self.assertEqual(str(cm.exception),'sampler = None is not allowed, expected sobol, latin')

    def test_tell(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 2,
            lower_bound = {'a':-10, 'b':-10},
            upper_bound = {'a':10, 'b':10},
            sampler = "sobol"
            )
        solver = SamplerSolver(solver_settings=solver_settings)
        solver.tell(pd.DataFrame([1,2]))
        self.assertTrue(solver.completed)

    def test_ask_sobol_dim(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = {'a':1, 'b':2, 'c':3},
            upper_bound = {'a':4, 'b':5, 'c':6},
            sampler = "sobol"
            )
        solver = SamplerSolver(solver_settings=solver_settings)

        self.assertEqual(solver.ask().shape,
            (solver.solver_settings.number_of_samples,
                len(solver.solver_settings.lower_bound)
                )
            )

    def test_ask_sobol_range(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = {'a':1, 'b':2, 'c':3},
            upper_bound = {'a':4, 'b':5, 'c':6},
            sampler = "sobol"
            )
        solver = SamplerSolver(solver_settings=solver_settings)
        pop = solver.ask()
        self.assertTrue(np.all((pop >= solver_settings.lower_bound) & (pop <= solver_settings.upper_bound)))

    def test_ask_bounds_transform(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 10000,
            lower_bound = {'a':1e-5, 'b':1, 'c':1},
            upper_bound = {'a':1e-2, 'b':10, 'c':10000},
            sampler = "sobol"
            )
        solver = SamplerSolver(solver_settings=solver_settings)
        self.assertListEqual(list(solver.scaler.log_scaler.log_columns), ['a', 'c'])

    def test_ask_latin_dim(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = {'a':1, 'b':2, 'c':3},
            upper_bound = {'a':4, 'b':5, 'c':6},
            sampler = "latin"
            )
        solver = SamplerSolver(solver_settings=solver_settings)
        self.assertEqual(solver.ask().shape,
            (solver.solver_settings.number_of_samples,
                len(solver.solver_settings.lower_bound)
                )
            )

    def test_ask_latin_range(self):
        solver_settings = SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = {'a':1, 'b':2, 'c':3},
            upper_bound = {'a':4, 'b':5, 'c':6},
            sampler = "latin"
            )
        solver = SamplerSolver(solver_settings=solver_settings)
        pop = solver.ask()
        self.assertTrue(np.all((pop >= solver_settings.lower_bound) & (pop <= solver_settings.upper_bound)))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSampler)
    unittest.TextTestRunner(verbosity=2).run(suite)
