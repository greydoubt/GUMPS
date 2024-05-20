# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the adaptive sampler application"

import unittest

import pandas as pd

from gumps.apps.adaptive_sampler import (AdaptiveSamplerApp,
                                         AdaptiveSamplerAppParameters)
from gumps.apps.parametric_sweep import ParametricSweepApp
from gumps.solvers.adaptive_solver import AdaptiveSamplerParameters
from gumps.solvers.regressors.gaussian_regressor import GaussianRegressor
from gumps.solvers.regressors.regression_solver import RegressionParameters
from gumps.solvers.sampler import SamplerSolverParameters
from gumps.studies.batch_sphere_study import BatchLineStudy
from gumps.solvers.simple_solver import SimpleSolver
from gumps.studies.batch_study import BatchStudyMultiProcess
from gumps.kernels.sphere_kernel import SphereKernelNanException
from gumps.studies.study import SimulationStudy
from gumps.common.parallel import Parallel


def get_total(frame:pd.DataFrame) -> pd.DataFrame:
    "processing function to get the total from the dataframe"
    return frame['total'].to_frame()

def pre_processing_function(input_data:pd.DataFrame) -> pd.DataFrame:
    "pre processing function to get the total from the dataframe"
    input_data_processed = input_data
    return pd.DataFrame(input_data_processed)

def exception_handler(function, x, e):
    return None

def get_total_dict(frame:pd.DataFrame|None) -> dict|None:
    if frame is None:
        return None
    return {'total': frame.total[0]}

class TestAdaptiveSamplerApp(unittest.TestCase):
    "test the adaptive sampler"

    def setUp(self):
        "create test setup to initialize the regressor with data"
        parameters = SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = {'x_0':-32, 'x_1':-32, 'x_2':-32, 'x_3':-32},
            upper_bound = {'x_0':32, 'x_1':32, 'x_2':32, 'x_3':32},
            sampler = "sobol"
            )
        model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}

        batch = BatchLineStudy(model_variables=model_variables,)

        app = ParametricSweepApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)
        app.run()

        self.regression_parameters = RegressionParameters(input_data=app.factors,
            output_data=app.responses)
        self.regressor = GaussianRegressor(self.regression_parameters)
        self.regressor.fit()
        self.batch = batch

    def test_init(self):
        "test the initialize method"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                 batch=self.batch,
                                 processing_function=None)

        self.assertIsInstance(app.batch, BatchLineStudy)
        self.assertIsInstance(app.parameters, AdaptiveSamplerAppParameters)

    def test_update_results_none(self):
        "test the update results method"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                 batch=self.batch,
                                 processing_function=get_total)

        new_inputs = pd.DataFrame({'x_0': [1, 2, 3, 4, 5], 'x_1': [1, 2, 3, 4, 5]})
        new_outputs = pd.DataFrame({'total': [1, 2, 3, 4, 5]})

        app.update_results(new_inputs, new_outputs)

        self.assertIsInstance(app.new_inputs, pd.DataFrame)
        self.assertIsInstance(app.new_outputs, pd.DataFrame)
        self.assertTupleEqual(app.new_inputs.shape, (5, 2))
        self.assertTupleEqual(app.new_outputs.shape, (5, 1))

    def test_update_results(self):
        "test the update results method with existing data"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                 batch=self.batch,
                                 processing_function=get_total)

        new_inputs = pd.DataFrame({'x_0': [1, 2, 3, 4, 5], 'x_1': [1, 2, 3, 4, 5]})
        new_outputs = pd.DataFrame({'total': [1, 2, 3, 4, 5]})

        app.update_results(new_inputs, new_outputs)

        new_inputs = pd.DataFrame({'x_0': [6, 7, 8, 9, 10], 'x_1': [6, 7, 8, 9, 10]})
        new_outputs = pd.DataFrame({'total': [6, 7, 8, 9, 10]})

        app.update_results(new_inputs, new_outputs)

        self.assertIsInstance(app.new_inputs, pd.DataFrame)
        self.assertIsInstance(app.new_outputs, pd.DataFrame)
        self.assertTupleEqual(app.new_inputs.shape, (10, 2))
        self.assertTupleEqual(app.new_outputs.shape, (10, 1))


    def test_run_user_continue(self):
        "test the run user continue method"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                 batch=self.batch,
                                 processing_function=get_total)

        self.assertTrue(app.run_user_continue())

    def test_run_user_continue_false(self):
        "test the run user continue method returns false"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters,
            user_continue=lambda x, y: False
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                 batch=self.batch,
                                 processing_function=get_total)

        self.assertFalse(app.run_user_continue())

    def test_run(self):
        "test the run method doesn't raise an error"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                 batch=self.batch,
                                 processing_function=get_total)
        app.run()

    def test_regressor(self):
        "test the the regressor has new data added to it after running"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                 batch=self.batch,
                                 processing_function=get_total)
        app.run()

        self.assertTrue(len(app.parameters.regressor.parameters.input_data) in {104, 105})

    def test_pre_processing(self):
        "test to check the adaptive sampler pre processing"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        app = AdaptiveSamplerApp(parameters=parameters,
                                    batch=self.batch,
                                    processing_function=get_total,
                                    pre_processing_function=pre_processing_function)
        app.run()


class TestAdaptiveSamplerExceptionNan(unittest.TestCase):

    def setUp(self):
        "create test setup to initialize the regressor with data"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_2':4, 'x_3':5, 'nan_lower_bound' : -1, 'nan_upper_bound':-0.5, 'n':4,
                   'exception_lower_bound':-1, 'exception_upper_bound':-0.5} #variables we want to keep fixed
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernelNanException(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        parameters = SamplerSolverParameters(
            number_of_samples = 100,
            lower_bound = {'x_0':-32, 'x_1':-32, 'x_2':-32, 'x_3':-32, 'nan_trigger':0, 'exception_trigger':0},
            upper_bound = {'x_0':32, 'x_1':32, 'x_2':32, 'x_3':32, 'nan_trigger':1, 'exception_trigger':0},
            sampler = "sobol"
            )

        app = ParametricSweepApp(parameters=parameters,
            processing_function=get_total_dict,
            directory=None,
            batch=batch)
        app.run()

        self.regression_parameters = RegressionParameters(input_data=app.factors,
            output_data=app.responses)
        self.regressor = GaussianRegressor(self.regression_parameters)
        self.regressor.fit()
        self.batch = batch


    def test_run_nan(self):
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        self.batch.study.solver._problem = {'x_0': 1.5, 'x_2':4, 'x_3':5,
                                            'nan_lower_bound' : 0.25, 'nan_upper_bound':0.75, 'n':4,
                                            'exception_lower_bound':-1, 'exception_upper_bound':-0.5}

        app = AdaptiveSamplerApp(parameters=parameters,
                                    batch=self.batch,
                                    processing_function=get_total_dict,
                                    pre_processing_function=pre_processing_function)
        app.run()


        self.assertIsInstance(app.new_inputs, pd.DataFrame)
        self.assertIsInstance(app.new_outputs, pd.DataFrame)
        self.assertTupleEqual(app.new_inputs.shape, (5, 6))
        self.assertTupleEqual(app.new_outputs.shape, (5, 1))


    def test_run_nan_exception(self):
        "test a mixture of nan and exceptions is filtered"
        sampler_parameters = AdaptiveSamplerParameters(
            points_to_add=5,
            batch_size=1,
            max_iterations=10,
            population_size=10
        )

        parameters = AdaptiveSamplerAppParameters(
            regressor=self.regressor,
            sampler_parameters = sampler_parameters
            )

        self.batch.study.solver._problem = {'x_0': 1.5, 'x_2':4, 'x_3':5,
                                            'nan_lower_bound' : 0.25, 'nan_upper_bound':0.75, 'n':4,
                                            'exception_lower_bound':0.25, 'exception_upper_bound':0.75}
        self.batch.exception_handler = exception_handler

        app = AdaptiveSamplerApp(parameters=parameters,
                                    batch=self.batch,
                                    processing_function=get_total_dict,
                                    pre_processing_function=pre_processing_function)
        app.run()


        self.assertIsInstance(app.new_inputs, pd.DataFrame)
        self.assertIsInstance(app.new_outputs, pd.DataFrame)
        self.assertTupleEqual(app.new_inputs.shape, (5, 6))
        self.assertTupleEqual(app.new_outputs.shape, (5, 1))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveSamplerApp)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suites = unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveSamplerExceptionNan)
    unittest.TextTestRunner(verbosity=2).run(suites)
