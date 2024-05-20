# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Integration test suite. The functions here are used as examples and demonstrate interfaces."

import unittest

import numpy as np
import pandas as pd
import scipy.stats

import gumps.apps.adaptive_sampler
import gumps.apps.monte_carlo
import gumps.apps.parametric_sweep
import gumps.solvers.monte_carlo_solver
import gumps.solvers.regressors.regression_solver
import gumps.solvers.regressors.trivial_regressor
import gumps.solvers.sampler
import gumps.studies.batch_sphere_study
import gumps.utilities.smoothing
from gumps.loss.loss import SumSquaredErrorBatch


class TestIntegration(unittest.TestCase):
    "integration tests that are used as examples and demonstrate interfaces"

    def test_sample_solver(self):
        solver_settings = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = [1, 2, 3, 4],
            upper_bound = [5, 6, 7, 8],
            sampler = "sobol"
            )
        solver =  gumps.solvers.sampler.SamplerSolver(solver_settings=solver_settings)
        model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        sse_loss  = SumSquaredErrorBatch(target=pd.DataFrame([{'total':0}]), weights=None)
        with gumps.studies.batch_sphere_study.BatchSphereStudy(model_variables=model_variables) as batch:
            while solver.has_next():
                pop = solver.ask()

                pop = pd.DataFrame(pop, columns=['x_0', 'x_1', 'x_2', 'x_3'])

                loss = batch.run(pop, sse_loss.run)

                solver.tell(loss.to_numpy())

    def test_regression(self):
        input_data = pd.DataFrame({'x1': np.linspace(0, 1, 100),  'x2': np.linspace(0, 1, 100)})
        output_data=  pd.DataFrame({'y1': input_data['x1'] + input_data['x2']})
        parameters = gumps.solvers.regressors.regression_solver.RegressionParameters(
            input_data=input_data,
            output_data=output_data
        )
        reg = gumps.solvers.regressors.trivial_regressor.TrivialLinearRegressor(parameters)
        reg.fit()

        loss = reg.error_metrics()

    def test_regression_sampler(self):
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = {'x_0':0, 'x_1':0, 'x_2':0},
            upper_bound = {'x_0':1, 'x_1':1, 'x_2':1},
            sampler = "sobol"
            )
        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0}

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def get_total(frame:pd.DataFrame):
            return pd.DataFrame({'total': frame.total})

        app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)
        app.run()

        parameters = gumps.solvers.regressors.regression_solver.RegressionParameters(
            input_data=app.factors,
            output_data=app.responses)
        reg = gumps.solvers.regressors.trivial_regressor.TrivialLinearRegressor(parameters)
        reg.fit()

        loss = reg.error_metrics()

    def test_smoothing(self):
        "create an intergration test to demonstrate how smoothing works"
        x = np.linspace(0, 4*np.pi, 1000)
        y = np.sin(x)
        y_noise = y * scipy.stats.norm(1, 0.01).rvs(size=len(x)) + scipy.stats.norm(0, 0.01).rvs(size=len(x))

        params = gumps.utilities.smoothing.SmoothingInput(pd.DataFrame({'x': x, 'y': y_noise}), 'x', 'y')
        smooth = gumps.utilities.smoothing.Smoothing(params)

        output = smooth.signal()

        before_noise_removal = np.linalg.norm(y_noise - y)
        after_noise_removal = np.linalg.norm(output.signal - y)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    unittest.TextTestRunner(verbosity=2).run(suite)
