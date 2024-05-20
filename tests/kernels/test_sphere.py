# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""This module has tests that use the Sphere Kernel"""

import unittest

from gumps.solvers.simple_solver import SimpleSolver
from gumps.kernels.sphere_kernel import SphereKernel
from gumps.studies.study import SimulationStudy

class TestSphere(unittest.TestCase):
    "test the new solver interface"

    def test_sphere_offset(self):
        "test that an offset from the correct sphere point calculates correctly"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_1':1.6, 'x_2': 0.7, 'x_3': 1.1, 'n': 4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        result = study.run()

        self.assertAlmostEqual(result.total, 5.77)

    def test_sphere_correct(self):
        "test that the correct spot for the sphere calculates correctly"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 0.1, 'x_1':0.2, 'x_2': 0.3, 'x_3': -0.2, 'n': 4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        result = study.run()

        self.assertAlmostEqual(result.total, 0.00)



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSphere)
    unittest.TextTestRunner(verbosity=2).run(suite)