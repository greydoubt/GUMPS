# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Test the parametric sweep app"

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import gumps.apps.parametric_sweep
import gumps.solvers.sampler
from gumps.common.hdf5 import H5
from gumps.common.parallel import Parallel
from gumps.kernels.sphere_kernel import SphereKernel, SphereKernelNanException
from gumps.kernels.ackley_complete_kernel import AckleyCompleteKernelAlternate
from gumps.solvers.simple_solver import SimpleSolver
from gumps.studies.batch_study import BatchStudyMultiProcess
from gumps.studies.study import SimulationStudy, SimpleSimulationStudy

def exception_handler(function, x, e):
    return None

def get_total(frame:pd.DataFrame|None) -> dict|None:
    if frame is None:
        return None
    return {'total': frame.total[0]}

def get_total_x(frame:pd.DataFrame|None) -> dict|None:
    if frame is None:
        return None
    return {'total': frame.x_0[0]}

class TestParametricSweep(unittest.TestCase):
    "integration tests that are used as examples and demonstrate interfaces"

    def test_parametric_sweep_pre_processing(self):
        "integration test for parametric sweep"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 1000,
            lower_bound = {'x1':-32, 'x2':-32, 'x3':-32, 'x4':-32, 'a':19, 'b':0.1, 'c':np.pi},
            upper_bound = {'x1':32, 'x2':32, 'x3':32, 'x4':32, 'a':20, 'b':0.2, 'c':2*np.pi},
            sampler = "sobol"
            )

        kernel = AckleyCompleteKernelAlternate()

        problem = {'x' : np.array([1,2]),'a' : 1,'b' : 2 ,'c' : 1}

        study = SimpleSimulationStudy(problem=problem, kernel=kernel)
        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        def pre_processing_function(input_data: pd.DataFrame):
            "pre-process the data"
            input_data_processed = {}
            input_data_processed['x'] = list(np.column_stack([input_data.x1, input_data.x2, input_data.x3, input_data.x4]))
            input_data_processed['a'] = input_data.a
            input_data_processed['b'] = input_data.b
            input_data_processed['c'] = input_data.c
            return pd.DataFrame(input_data_processed)

        app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
            processing_function=get_total,
            pre_processing_function=pre_processing_function,
            directory=None,
            batch=batch)
        app.run()

    def test_parametric_sweep(self):
        "integration test for parametric sweep"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 10,
            lower_bound = {'x_1':1, 'x_2':2, 'x_3':3},
            upper_bound = {'x_1':5, 'x_2':6, 'x_3':7},
            sampler = "sobol"
            )
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)
        app.run()

    def test_run_nan(self):
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 10,
            lower_bound = {'x_1':1, 'x_2':2, 'x_3':3, 'nan_trigger':0},
            upper_bound = {'x_1':5, 'x_2':6, 'x_3':7, 'nan_trigger':1},
            sampler = "sobol"
            )

        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'nan_lower_bound' : 0, 'nan_upper_bound':1, 'n':4,
                   'exception_lower_bound':-1, 'exception_upper_bound':-0.5} #variables we want to keep fixed
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernelNanException(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(app.factors.empty)
        self.assertTrue(app.responses.empty)

    def test_run_nan_exception(self):
        "test a mixture of nan and exceptions is filtered"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 10,
            lower_bound = {'x_1':1, 'x_2':2, 'x_3':3, 'nan_trigger':0, 'exception_trigger':0},
            upper_bound = {'x_1':5, 'x_2':6, 'x_3':7, 'nan_trigger':1, 'exception_trigger':1},
            sampler = "sobol"
            )

        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'nan_lower_bound' : 0.25, 'nan_upper_bound':0.75, 'n':4,
                   'exception_lower_bound':0.25, 'exception_upper_bound':0.75} #variables we want to keep fixed
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernelNanException(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(len(app.factors) == 3)
        self.assertTrue(len(app.responses) == 3)

    def test_parametric_sweep_save(self):
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 10,
            lower_bound = {'x_1':1, 'x_2':2, 'x_3':3},
            upper_bound = {'x_1':5, 'x_2':6, 'x_3':7},
            sampler = "sobol"
            )
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
                processing_function=get_total,
                directory=directory,
                batch=batch)
            app.run()

            app.save_data_hdf5()

            h5 = H5( (directory / "data.h5").as_posix() )
            h5.load()

            np.testing.assert_equal(h5.root.factors, app.factors)
            np.testing.assert_equal(h5.root.responses, app.responses)
            np.testing.assert_equal([x.decode() for x in h5.root.factor_names], list(app.parameters.lower_bound.keys()))

    def test_save_data_hdf5_exception(self):
        "test that hdf5 raises an exception if used before the simulation is run"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 10,
            lower_bound = {'x_1':1, 'x_2':2, 'x_3':3},
            upper_bound = {'x_1':5, 'x_2':6, 'x_3':7},
            sampler = "sobol"
            )
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
                processing_function=get_total,
                directory=directory,
                batch=batch)

            with self.assertRaises(RuntimeError):
                app.save_data_hdf5()

    def test_get_answer_exception(self):
        "test that answer raises an exception if used before the simulation is run"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 10,
            lower_bound = {'x_1':1, 'x_2':2, 'x_3':3},
            upper_bound = {'x_1':5, 'x_2':6, 'x_3':7},
            sampler = "sobol"
            )
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
            processing_function=get_total,
            batch=batch)

        with self.assertRaises(RuntimeError):
            app.answer()

    def test_parametric_sweep_fixed_values(self):
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 10,
            lower_bound = {'x_1':1, 'x_2':2, 'x_3':3},
            upper_bound = {'x_1':5, 'x_2':6, 'x_3':7},
            sampler = "sobol"
            )
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'n':4} #variables we want to keep fixed
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=2)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        app = gumps.apps.parametric_sweep.ParametricSweepApp(parameters=parameters,
            processing_function=get_total_x,
            directory=None,
            batch=batch)
        app.run()
        test = app.answer()
        correct = pd.DataFrame({'total': [1.5 for _ in range(parameters.number_of_samples)]})
        pd.testing.assert_frame_equal(test, correct)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParametricSweep)
    unittest.TextTestRunner(verbosity=2).run(suite)
