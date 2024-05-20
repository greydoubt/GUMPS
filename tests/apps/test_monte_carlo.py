# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the monte carlo application"

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import tempfile
from pathlib import Path

import gumps.apps.monte_carlo
import gumps.solvers.monte_carlo_solver
import gumps.studies.batch_sphere_study
from gumps.common.hdf5 import H5
from gumps.solvers.simple_solver import SimpleSolver
from gumps.studies.batch_study import BatchStudyMultiProcess
from gumps.kernels.sphere_kernel import SphereKernelNanException
from gumps.studies.study import SimulationStudy
from gumps.common.parallel import Parallel

def exception_handler(function, x, e):
    return None

def processing_function(frame: pd.DataFrame|None) -> dict|None:
    "process the dataframe for the loss function"
    if frame is None:
        return None
    return {'total': frame.total[0]}

class TestMonteCarloApp(unittest.TestCase):
    "test the monte carlo app"

    def test_monte_carlo(self):
        "integration test for monte carlo solver"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.05],
                window=10, tolerance=1e-2, min_steps=3, sampler_seed=0, sampler_scramble=False)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}
        diffs = [var.replace('a', 'd') for var in model_variables]

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame[diffs])

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            directory=None,
            batch=batch)
        app.run()

        answer = app.answer().to_numpy()

        correct = np.array([dist.ppf(parameters.target_probability) for dist in distributions.values()]).T.flatten()

        np.testing.assert_allclose(answer, correct, rtol=1e-1, atol=0)


    def test_monte_carlo_nan(self):
        "integration test for monte carlo solver for proper handling of nans in the batch study"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.05],
                window=10, tolerance=1e-2, min_steps=3, sampler_seed=0, sampler_scramble=False)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}
        diffs = [var.replace('a', 'd') for var in model_variables]

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            frame = pd.DataFrame(frame[diffs])
            frame.loc[frame.sample(frac=0.005).index] = np.nan
            return frame

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            directory=None,
            batch=batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(len(cm.output) > 0)

        answer = app.answer().to_numpy()

        correct = np.array([dist.ppf(parameters.target_probability) for dist in distributions.values()]).T.flatten()

        np.testing.assert_allclose(answer, correct, rtol=1e-1, atol=0)


    def test_monte_carlo_nan_exception(self):
        "integration test for monte carlo solver for proper handling of nans in the batch study"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_2':4, 'x_3':5, 'nan_lower_bound' : 0.25, 'nan_upper_bound':0.75, 'n':4,
                   'exception_lower_bound':0.25, 'exception_upper_bound':0.75} #variables we want to keep fixed
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernelNanException(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2),
                'nan_trigger':scipy.stats.uniform(0, 1), 'exception_trigger':scipy.stats.uniform(0, 1)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.05],
                window=10, tolerance=1e-2, min_steps=3, sampler_seed=0, sampler_scramble=False)

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            directory=None,
            batch=batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(len(cm.output) > 0)

        answer = app.answer()

        self.assertIsInstance(answer, pd.Series)
        self.assertTrue('total_0.05' in answer)

    def test_monte_carlo_multiple_prob(self):
        "integration test for monte carlo solver"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.9, 0.933],
                window=10, tolerance=1e-2, min_steps=3, sampler_seed=0, sampler_scramble=False)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}
        diffs = [var.replace('a', 'd') for var in model_variables]

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame[diffs])

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            directory=None,
            batch=batch)
        app.run()

        answer = app.answer().to_numpy()

        correct = np.array([dist.ppf(parameters.target_probability) for dist in distributions.values()]).T.flatten()

        np.testing.assert_allclose(answer, correct, rtol=1e-1, atol=0)


    def test_monte_carlo_scalar(self):
        "integration test for monte carlo solver"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.9],
                window=10, tolerance=1e-2, min_steps=3, sampler_seed=0, sampler_scramble=False)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame["total"])

        app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
            processing_function=processing_function,
            directory=None,
            batch=batch)
        app.run()

        answer = app.answer()

        self.assertEqual(len(answer), 1)


    def test_monte_carlo_plotting_scalar(self):
        "test that plotting works for the monte carlo solver"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.9],
                window=10, tolerance=1e-2, min_steps=3, sampler_seed=0, sampler_scramble=False)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame["total"])

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
                processing_function=processing_function,
                directory=Path(directory),
                batch=batch)
            app.run()

            app.create_plots()

            self.assertEqual(len(list(Path(directory).glob("*.png"))), 2)

    def test_monte_carlo_plotting_multi(self):
        "test that plotting works for the monte carlo solver"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.1, 0.9, 0.933],
                window=10, tolerance=1e-2, min_steps=50, sampler_seed=0, sampler_scramble=False, runnable_batch_size=1000)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}
        diffs = [var.replace('a', 'd') for var in model_variables]

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame[diffs])

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
                processing_function=processing_function,
                directory=Path(directory),
                batch=batch)
            app.run()

            app.create_plots()

            self.assertEqual(len(list(Path(directory).glob("*.png"))), 5)


    def test_monte_carlo_plotting_multi_directory_none(self):
        "test that plotting works for the monte carlo solver"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.1, 0.9, 0.933],
                window=2, tolerance=1e-2, min_steps=5, sampler_seed=0, sampler_scramble=False, runnable_batch_size=1000)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}
        diffs = [var.replace('a', 'd') for var in model_variables]

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame[diffs])

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
                processing_function=processing_function,
                directory=None,
                batch=batch)
            app.run()

            app.create_plots()


    def test_monte_carlo_save(self):
        "test that saving works for the monte carlo solver"
        distributions = {'x_0':scipy.stats.uniform(0.0, 1), 'x_1':scipy.stats.norm(0, 1),
                'x_2':scipy.stats.uniform(-1, 2), 'x_3':scipy.stats.norm(1, 1e-2)}

        parameters = gumps.solvers.monte_carlo_solver.MonteCarloParameters(variable_distributions=distributions, target_probability=[0.9, 0.5],
                window=10, tolerance=1e-2, min_steps=3, sampler_seed=0, sampler_scramble=False)

        model_variables = {'a_0': 0.0, 'a_1':0.0, 'a_2':0, 'a_3':0}

        batch = gumps.studies.batch_sphere_study.BatchLineStudy(model_variables=model_variables)

        def processing_function(frame: pd.DataFrame):
            "process the dataframe for the loss function"
            return pd.DataFrame(frame["total"])

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.monte_carlo.MonteCarloApp(parameters=parameters,
                processing_function=processing_function,
                directory=Path(directory),
                batch=batch)
            app.run()

            app.save_data_hdf5()

            h5 = H5(directory / "data.h5")
            h5.load()

            np.testing.assert_equal(h5.root.chain, app.chain)
            np.testing.assert_equal(h5.root.scores, app.scores)
            np.testing.assert_equal(h5.root.probability, app.parameters.target_probability)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMonteCarloApp)
    unittest.TextTestRunner(verbosity=2).run(suite)
