# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Test module for the Ackley kernel."

import unittest

import numpy as np
import pandas as pd

from gumps.kernels.time_kernel import TimeKernelExample, SimpleJupyterTimeKernel, SimpleTimeKernelState, TimeKernelState
from gumps.studies.study import SimulationStudy
from gumps.solvers.iterative_solver import IterativeSolver, IterativeSolverParameters

class TestTimeKernelExample(unittest.TestCase):
    "test the time kernel example"

    def test_get_states(self):
        "test the ackley first term kernel"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        kernel = TimeKernelExample(model_variables=model_variables)
        self.assertEqual(kernel.get_state_class(), TimeKernelState)

    def test_user_defined_function(self):
        "test the ackley first term with a user defined function"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':1}
        kernel = TimeKernelExample(model_variables=model_variables)

        states = kernel.f(problem)

        self.assertAlmostEqual(states.ta, problem['a'] * problem['time'])
        self.assertAlmostEqual(states.tb, problem['b'] * problem['time'])
        self.assertAlmostEqual(states.tc, problem['c'] * problem['time'])
        self.assertAlmostEqual(states.td, problem['d'] * problem['time'])

    def test_study(self):
        "test the ackley first term with a study"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)
        study.run()

        df = study.state_frame()

        time_series = np.linspace(0, 10, 100)
        ta = time_series * problem['a']
        tb = time_series * problem['b']
        tc = time_series * problem['c']
        td = time_series * problem['d']

        df_correct = pd.DataFrame({'ta':ta, 'tb':tb, 'tc':tc, 'td':td})
        pd.testing.assert_frame_equal(df[['ta','tb', 'tc', 'td']], df_correct)

class TestSimpleJupyterTimeKernel(unittest.TestCase):
    "test the time kernel example"

    def test_get_state_class(self):
        "test the ackley first term kernel"
        model_variables = {}
        kernel = SimpleJupyterTimeKernel(model_variables=model_variables)
        self.assertEqual(kernel.get_state_class(), SimpleTimeKernelState)

    def test_user_defined_function(self):
        "test the ackley first term with a user defined function"
        model_variables = {}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'time':1}
        kernel = SimpleJupyterTimeKernel(model_variables=model_variables)

        states = kernel.f(problem)

        self.assertAlmostEqual(states.c1, problem['a'] * np.sin(problem['time']) + problem['c'])
        self.assertAlmostEqual(states.c2, problem['b'] * np.cos(problem['time']) + problem['c'])
        self.assertAlmostEqual(states.c3, problem['a'] * np.sin(problem['time']) * problem['b'] * np.cos(problem['time']) + problem['c'])

    def test_study(self):
        "test the ackley first term with a study"
        model_variables = {}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = SimpleJupyterTimeKernel(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)
        study.run()

        df = study.state_frame()

        time_series = np.linspace(0, 10, 100)
        c1 = problem['a'] * np.sin(time_series) + problem['c']
        c2 = problem['b'] * np.cos(time_series) + problem['c']
        c3 = problem['a'] * np.sin(time_series) * problem['b'] * np.cos(time_series) + problem['c']

        df_correct = pd.DataFrame({'c1':c1, 'c2':c2, 'c3':c3})
        pd.testing.assert_frame_equal(df[['c1','c2', 'c3']], df_correct)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTimeKernelExample)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSimpleJupyterTimeKernel)
    unittest.TextTestRunner(verbosity=2).run(suite)
