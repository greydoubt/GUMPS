# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from typing import Callable

from gumps.solvers import AbstractSolver

class TestSolver(unittest.TestCase):
    "test the new solver interface"

    def test_create_abstract(self):
        "test that creating an abstract class fails"
        with self.assertRaises(TypeError):
            AbstractSolver({}, {})

    def test_create_position(self):
        "test that creating using positional arguments fails"
        class Foo(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                pass

        with self.assertRaises(TypeError):
            Foo({}, {})

    def test_missing_solve(self):
        "test that creation fails if solve is not defined"
        class Foo(AbstractSolver):
            pass

        with self.assertRaises(TypeError):
            Foo({}, {})

    def test_creation(self):
        "test that a subclass can be created and variables set"
        class Foo(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                pass

        problem = {'a':1, 'b':2}
        solver_settings = {'c':3, 'd':4}
        foo = Foo(problem=problem, solver_settings=solver_settings)

        self.assertEqual(foo.problem, problem)
        self.assertEqual(foo.solver_settings, solver_settings)

    def test_log_problem(self):
        "the that the problem can be logged"
        class Foo(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                pass

        problem = {'a':1, 'b':2}
        solver_settings = {'c':3, 'd':4}
        foo = Foo(problem=problem, solver_settings=solver_settings)

        with self.assertLogs('gumps.solvers.solver', level='DEBUG') as cm:
            foo.problem
        self.assertEqual(cm.output, ['DEBUG:gumps.solvers.solver:making a copy of self._problem'])

    def test_solve(self):
        class TrivialSolver(AbstractSolver):
            def solve(self, f, save_function):
                return f(self.problem)
        class Bar():
            cls_prop = 3
            def f(self, dict):
                return dict['number'] * self.cls_prop
        testSolver = TrivialSolver(problem={'number' : 3}, solver_settings=None)
        testFunction = Bar()
        value = testSolver.solve(testFunction.f, None)
        self.assertEqual(value, 9)

    def test_new_solver(self):
        "test creating a new solver"
        class Foo(AbstractSolver):
            def solve(self, f:Callable, save_function: Callable):
                pass

        problem = {'a':1, 'b':2}
        solver_settings = {'c':3, 'd':4}
        solver = Foo(problem=problem, solver_settings=solver_settings)

        new_solver = solver.new_solver({'a':3})

        self.assertDictEqual(new_solver._problem, {'a':3, 'b':2})


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)