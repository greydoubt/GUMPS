# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from typing import Callable
import pandas as pd
import numpy as np
import attrs

import gumps.studies
from gumps.kernels import AbstractKernel
from gumps.solvers import AbstractSolver
from gumps.common import IllDefinedException
import gumps.solvers.simple_solver

@attrs.define
class KernelStateList:
    a:int
    t: float
    b: list[float] = [1.0,2.0,3.0]
    c: int| float = None

@attrs.define
class KernelState:
    a:int
    t: float
    b: int = 2
    c: int| float = None

class TestStudy(unittest.TestCase):
    "test the new study interface"

    def test_study_problem_mismatch(self):
        "test that the study raises an exception when the problem variables are not a subset of the models"
        class Kernel(AbstractKernel):
            def initialize(self):
                self.model = {'a':1, 'b':[1.0,2.0,3.0], 'c':None}
            def user_defined_function(self, KernelState: object):
                pass
            def get_state_class(self) -> KernelStateList:
                return KernelStateList

        class MySolver(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                pass

        problem = {'a':1, 'b':2, 'e':5}
        solver_settings = {'c':3, 'd':4}

        model_variables = {'a':1, 'b':2, 'c':3}



        solver = MySolver(problem=problem, solver_settings=solver_settings)
        kernel = Kernel(model_variables=model_variables)

        with self.assertRaises(IllDefinedException):
            gumps.studies.SimulationStudy(solver, kernel)

    def test_simple_study(self):
        "test a simple study"
        class Kernel(AbstractKernel):
            def initialize(self):
                self.model = {'a':1, 'b':[1.0,2.0,3.0], 'c':None, 't':0}
            def user_defined_function(self, variables: KernelStateList):
                variables.c = variables.a * variables.t
            def get_state_class(self) -> KernelStateList:
                return KernelStateList

        problem = {'a':1, 't':0}

        model_variables = {'a':1, 'b':2, 'c':3}

        kernel = Kernel(model_variables=model_variables)

        study = gumps.studies.SimpleSimulationStudy(problem, kernel)

        self.assertTrue(study.kernel is kernel)
        self.assertTrue(isinstance(study.solver, gumps.solvers.simple_solver.SimpleSolver))

    def test_study_problem(self):
        "test that the study works with save, check_problem, and run"
        class Kernel(AbstractKernel):
            def initialize(self):
                self.model = {'a':1, 'b':[1.0,2.0,3.0], 'c':None, 't':0}
            def user_defined_function(self, variables: KernelStateList):
                variables.c = variables.a * variables.t
            def get_state_class(self) -> KernelStateList:
                return KernelStateList

        class MySolver(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                for i in range(10):
                    problem = self.problem
                    problem['t'] = i
                    state = f(problem)
                    save_function(state)

        problem = {'a':1, 't':0}
        solver_settings = {'c':3, 'd':4}

        model_variables = {'a':1, 'b':2, 'c':3}



        solver = MySolver(problem=problem, solver_settings=solver_settings)
        kernel = Kernel(model_variables=model_variables)

        with self.subTest("check_problem"):
            study = gumps.studies.SimulationStudy(solver, kernel)

        with self.subTest("check run"):
            study.run()

        with self.subTest("test length"):
            self.assertEqual(len(study.states), 10)

        for state in study.states:
            with self.subTest(f"t={state.t}"):
                self.assertEqual(state.c, state.a*state.t)

    def test_log_study_problem(self):
        "test that the study works with save, check_problem, and run"
        class Kernel(AbstractKernel):
            def initialize(self):
                self.model = {'a':1, 'b':[1.0,2.0,3.0], 'c':None, 't':0}
            def user_defined_function(self, variables: KernelStateList):
                variables.c = variables.a * variables.t
            def get_state_class(self) -> KernelStateList:
                return KernelStateList

        class MySolver(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                problem = self.problem
                problem['t'] = 0
                state = f(problem)
                save_function(state)

        problem = {'a':1, 't':0}
        solver_settings = {'c':3, 'd':4}

        model_variables = {'a':1, 'b':2, 'c':3}

        solver = MySolver(problem=problem, solver_settings=solver_settings)
        kernel = Kernel(model_variables=model_variables)

        with self.subTest("check_problem"):
            with self.assertLogs('gumps.studies.study', level='DEBUG') as cm:
                study = gumps.studies.SimulationStudy(solver, kernel)
            allowed_test = study.kernel.allowed_state
            problem_test = set(study.solver.problem)
            correct = [f'DEBUG:gumps.studies.study:study allowed variables: {allowed_test}  problem variables: {problem_test}']
            self.assertEqual(cm.output, correct)

        with self.subTest("check run"):
            with self.assertLogs('gumps.studies.study', level='DEBUG') as cm:
                study.run()
            correct = ['DEBUG:gumps.studies.study:running the study',
                'DEBUG:gumps.studies.study:saving the state  KernelStateList(a=1, t=0, b=[1.0, 2.0, 3.0], c=0)',
                'DEBUG:gumps.studies.study:study has completed with result: None'
                ]
            self.assertEqual(cm.output, correct)

    def test_study_run_data(self):
        "test that the study works with run_data"
        class Kernel(AbstractKernel):
            def initialize(self):
                self.model = {'a':1, 'b':2, 'c':None, 't':0}
            def user_defined_function(self, variables: KernelState):
                variables.c = variables.a * variables.t
            def get_state_class(self) -> KernelState:
                return KernelState

        class MySolver(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                for i in range(2):
                    problem = self.problem
                    problem['t'] = i
                    state = f(problem)
                    save_function(state)

        problem = {'a':1, 't':0}
        solver_settings = {'c':3, 'd':4}

        model_variables = {'a':1, 'b':2, 'c':3}

        solver = MySolver(problem=problem, solver_settings=solver_settings)
        kernel = Kernel(model_variables=model_variables)

        study = gumps.studies.SimulationStudy(solver, kernel)

        df = study.run_data({'a':2})

        correct = pd.DataFrame([{'a':2, 't':0, 'b':2, 'c':0}, {'a':2, 't':1, 'b':2, 'c':2}])

        pd.testing.assert_frame_equal(df, correct)

class TestAbstractStudy(unittest.TestCase):
    "test the abstract study class"

    def test_abstract_study(self):
        "test that the abstract study class raises an exception when the abstract methods are called"
        with self.assertRaises(TypeError):
            gumps.studies.AbstractStudy()

class TestTestStudy(unittest.TestCase):
    "test the test study class"

    def test_creation(self):
        "test that the test study class raises an exception when the abstract methods are called"
        study = gumps.studies.TestStudy(5)

        self.assertEqual(study.x, 5)

    def test_run(self):
        "test that the test study class raises an exception when the abstract methods are called"
        study = gumps.studies.TestStudy(5)

        state = study.run()

        self.assertEqual(state.x, 5)
        self.assertEqual(state.y, 5)

    def test_run_exception(self):
        "test that the test study class raises an exception when the abstract methods are called"
        study = gumps.studies.TestStudy(-5)

        with self.assertRaises(Exception):
            study.run()

    def test_state_frame(self):
        "test that the test study class raises an exception when the abstract methods are called"
        study = gumps.studies.TestStudy(5)

        frame = study.state_frame()

        correct = pd.DataFrame([{'x':5, 'y':5}])

        pd.testing.assert_frame_equal(frame, correct)

    def test_run_data(self):
        "test that the test study class raises an exception when the abstract methods are called"
        study = gumps.studies.TestStudy(5)

        frame = study.run_data({'x':5})

        correct = pd.DataFrame([{'x':5, 'y':5}])

        pd.testing.assert_frame_equal(frame, correct)

    def test_run_data_exception(self):
        "test that the test study class raises an exception when the abstract methods are called"
        study = gumps.studies.TestStudy(5)

        with self.assertRaises(Exception):
            study.run_data({'x':-5})

    def test_run_data_nan(self):
        "test that the test study class raises an exception when the abstract methods are called"
        study = gumps.studies.TestStudy(5)

        frame = study.run_data({'x':20})

        correct = pd.DataFrame([{'x':20, 'y':np.nan}])

        pd.testing.assert_frame_equal(frame, correct)


class TestStudyTime(unittest.TestCase):

    def test_init(self):
        "test that the test study class raises an exception when the abstract methods are called"
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        self.assertEqual(study.model_variables, model_variables)
        self.assertEqual(study.problem, problem)
        self.assertTrue(isinstance(study.time_series, np.ndarray))

    def test_run(self):
        "test that the test study class raises an exception when the abstract methods are called"
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        frame = study.run()

        correct = pd.DataFrame({'time':[0.0, 1.0], 'ta':[1.0, 2.0], 'tb':[2.0, 4.0], 'tc':[3.0, 6.0]})
        correct.set_index('time', inplace=True)

        pd.testing.assert_frame_equal(frame, correct)

    def test_state_frame(self):
        "test that the test study class raises an exception when the abstract methods are called"
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        study.run()

        frame = study.state_frame()

        correct = pd.DataFrame({'time':[0.0, 1.0], 'ta':[1.0, 2.0], 'tb':[2.0, 4.0], 'tc':[3.0, 6.0]})
        correct.set_index('time', inplace=True)

        pd.testing.assert_frame_equal(frame, correct)

    def test_state_frame_exception(self):
        "test that the test study class raises an exception when the abstract methods are called"
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        with self.assertRaises(RuntimeError):
            study.state_frame()


    def test_run_data(self):
        "test that the test study class raises an exception when the abstract methods are called"
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        frame = study.run_data({'a':1, 'b':2, 'c':3})

        correct = pd.DataFrame({'time':[0.0, 1.0], 'ta':[1.0, 2.0], 'tb':[2.0, 4.0], 'tc':[3.0, 6.0]})
        correct.set_index('time', inplace=True)

        pd.testing.assert_frame_equal(frame, correct)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStudy)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAbstractStudy)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestTestStudy)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestStudyTime)
    unittest.TextTestRunner(verbosity=2).run(suite)