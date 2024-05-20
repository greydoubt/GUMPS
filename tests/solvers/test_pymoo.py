# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pandas as pd

from gumps.solvers.pymoo_solvers import PyMooSolver, PyMooSolverParameters
from gumps.studies.batch_sphere_study import BatchSphereStudy
from gumps.loss.loss import SumSquaredErrorBatch

def check_algorithm(algorithm_name, objectives):
    "check an algorithm"
    model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
    solver_settings = PyMooSolverParameters(
        number_var=4,
        number_obj=len(objectives),
        lower_bound=[-10, -10, -10, -10],
        upper_bound=[10, 10, 10, 10],
        auto_transform=False,
        population_size=10,
        algorithm_name=algorithm_name,
        total_generations=2
    )

    solver = PyMooSolver(solver_settings=solver_settings)
    sse_loss  = SumSquaredErrorBatch(target=pd.DataFrame([objectives]), weights=None)
    with BatchSphereStudy(model_variables=model_variables) as batch:
        while solver.has_next():
            pop = solver.ask()

            pop = pd.DataFrame(pop, columns=['x_0', 'x_1', 'x_2', 'x_3'])

            loss = batch.run(pop, sse_loss.run)

            solver.tell(loss.to_numpy())

class TestPyMooStudy(unittest.TestCase):
    "test the pyMoo interface"

    def test_pymoo_nsga2_so(self):
        "test simple pymoo interface"
        check_algorithm('nsga2', {'total':0})

    def test_pymoo_rnsga2_so(self):
        "test simple pymoo interface"
        check_algorithm('rnsga2', {'total':0})

    def test_pymoo_nsga3_so(self):
        "test simple pymoo interface"
        check_algorithm('nsga3', {'total':0})

    def test_pymoo_unsga3_so(self):
        "test simple pymoo interface"
        check_algorithm('unsga3', {'total':0})

    def test_pymoo_rvea_so(self):
        "test simple pymoo interface"
        with self.assertRaises(ValueError):
            check_algorithm('rvea', {'total':0})

    def test_pymoo_smsemoa_so(self):
        "test simple pymoo interface"
        check_algorithm('smsemoa', {'total':0})

    def test_pymoo_cmaes_so(self):
        "test simple pymoo interface"
        check_algorithm('cmaes', {'total':0})

    def test_pymoo_nsga2_mo(self):
        "test simple pymoo interface"
        check_algorithm('nsga2', {'d_0':0, 'd_1':0, 'd_2':0, 'd_3':0})

    def test_pymoo_rnsga2_mo(self):
        "test simple pymoo interface"
        check_algorithm('rnsga2', {'d_0':0, 'd_1':0, 'd_2':0, 'd_3':0})

    def test_pymoo_nsga3_mo(self):
        "test simple pymoo interface"
        check_algorithm('nsga3', {'d_0':0, 'd_1':0, 'd_2':0, 'd_3':0})

    def test_pymoo_unsga3_mo(self):
        "test simple pymoo interface"
        check_algorithm('unsga3', {'d_0':0, 'd_1':0, 'd_2':0, 'd_3':0})

    def test_pymoo_rvea_mo(self):
        "test simple pymoo interface"
        check_algorithm('rvea', {'d_0':0, 'd_1':0, 'd_2':0, 'd_3':0})

    def test_pymoo_smsemoa_mo(self):
        "test simple pymoo interface"
        check_algorithm('smsemoa', {'d_0':0, 'd_1':0, 'd_2':0, 'd_3':0})

    def test_pymoo_cmaes_mo(self):
        "test simple pymoo interface"
        with self.assertRaises(ValueError):
            check_algorithm('cmaes', {'d_0':0, 'd_1':0, 'd_2':0, 'd_3':0})


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPyMooStudy)
    unittest.TextTestRunner(verbosity=2).run(suite)

