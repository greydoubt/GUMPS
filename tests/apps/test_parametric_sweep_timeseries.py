# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Test the parametric sweep app"

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import gumps.apps.parametric_sweep_timeseries
import gumps.solvers.sampler
from gumps.common.hdf5 import H5
from gumps.common.parallel import Parallel
from gumps.studies.study import SimulationStudy
from gumps.kernels.time_kernel import TimeKernelExample
from gumps.solvers.iterative_solver import IterativeSolver, IterativeSolverParameters
from gumps.studies.batch_time_study import BatchTimeStudyMultiProcess
import gumps.studies.batch_time_study_example

def exception_handler(function, x, e):
    return None

def pre_processing_function(frame:pd.DataFrame|None) -> pd.DataFrame|None:
    return frame

def get_total(frame:pd.DataFrame|None) -> pd.DataFrame|None:
    if frame is None:
        return None
    return frame.set_index("time")

def get_total_x(frame:pd.DataFrame|None) -> pd.DataFrame|None:
    if frame is None:
        return None
    return frame.set_index("time")

class TestParametricSweepTimeSeries(unittest.TestCase):
    "integration tests that are used as examples and demonstrate interfaces"

    def test_parametric_sweep(self):
        "integration test for parametric sweep"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)
        app.run()

    def test_run_streaming(self):
        "test the run streaming method with a temporary directory"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
                processing_function=get_total,
                directory=directory,
                batch=batch)
            app.run_streaming(save_interval_seconds=1)

            self.assertTrue((directory / "data.h5").exists())

    def test_run_streaming_exception_directory(self):
        "test that run streaming raises an exception if the directory is not set"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':-5, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : -2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)
        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)
        study = SimulationStudy(kernel=kernel, solver=solver)
        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            batch=batch)

        with self.assertRaises(RuntimeError):
            app.run_streaming(save_interval_seconds=1)

    def test_run_streaming_pre_processing(self):
        "test the run streaming method with a temporary directory and pre processing function"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':-4, 'b':-4, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : -2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)
        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)
        study = SimulationStudy(kernel=kernel, solver=solver)
        parallel = Parallel(poolsize=4)
        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
                processing_function=get_total,
                pre_processing_function=pre_processing_function,
                directory=directory,
                batch=batch)
            app.run_streaming(save_interval_seconds=1)

            self.assertTrue((directory / "data.h5").exists())

    def test_process_interval(self):
        "test the process interval method"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':-4, 'b':-4, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        factors = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6]})
        responses = [pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]}),
                    pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]}),
                    pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]})]
        responses = [x.set_index("time") for x in responses]

        batch = gumps.studies.batch_time_study_example.BatchTimeSeriesExample()

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
                processing_function=get_total,
                pre_processing_function=pre_processing_function,
                batch=batch)

        factors, responses = app.process_interval(factors, [0,1,2], responses)

        self.assertEqual(factors.shape, (3,2))
        self.assertEqual(responses[0].shape, (3,2))
        self.assertEqual(responses[1].shape, (3,2))
        self.assertEqual(responses[2].shape, (3,2))

    def test_process_interval_nan(self):
        "test the process interval method with nan values"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':-4, 'b':-4, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        factors = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6]})
        responses = [pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]}),
                    pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]}),
                    pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[np.nan,5,6]})]
        responses = [x.set_index("time") for x in responses]

        batch = gumps.studies.batch_time_study_example.BatchTimeSeriesExample()

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
                processing_function=get_total,
                pre_processing_function=pre_processing_function,
                batch=batch)

        factors, responses = app.process_interval(factors, [0,1,2], responses)

        self.assertEqual(factors.shape, (2,2))
        self.assertEqual(responses[0].shape, (3,2))
        self.assertEqual(responses[1].shape, (3,2))

    def test_run_nan(self):
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':-5, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(len(app.factors) == 4)
        self.assertTrue(len(app.responses.index) == 4)

    def test_run_nan_exception(self):
        "test a mixture of nan and exceptions is filtered"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':-4, 'b':-4, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            directory=None,
            batch=batch)

        with self.assertLogs('gumps.common.app_utils', level='WARNING') as cm:
            app.run()

        self.assertTrue(len(app.factors) == 2)
        self.assertTrue(len(app.responses.index) == 2)

    def test_parametric_sweep_save(self):
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
                processing_function=get_total,
                directory=directory,
                batch=batch)
            app.run()

            app.save_data_hdf5()

            h5 = H5( (directory / "data.h5").as_posix() )
            h5.load()

            np.testing.assert_equal(h5.root.factors, app.factors)
            np.testing.assert_equal(h5.root.responses, app.responses.to_array().values)
            np.testing.assert_equal([x.decode() for x in h5.root.factor_names], list(app.parameters.lower_bound.keys()))
            np.testing.assert_equal([x.decode() for x in h5.root.response_names], list(app.responses.keys()))

    def test_save_data_hdf5_exception(self):
        "test that hdf5 raises an exception if used before the simulation is run"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
                processing_function=get_total,
                directory=directory,
                batch=batch)

            with self.assertRaises(RuntimeError):
                app.save_data_hdf5()

    def test_get_answer_exception(self):
        "test that answer raises an exception if used before the simulation is run"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            batch=batch)

        with self.assertRaises(RuntimeError):
            app.answer()

    def test_get_answer(self):
        "test that answer after a simulation has run"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            batch=batch)

        app.run()
        data = app.answer()

        self.assertIsInstance(data, xr.Dataset)

    def test_update_hdf5_nodata(self):
        "make sure that no error is raised if the data is empty"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            batch=batch)

        with self.assertLogs('gumps.apps.parametric_sweep_timeseries', level='INFO') as cm:
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                app.directory = directory
                app.update_hdf5(factors=pd.DataFrame(), responses=[pd.DataFrame()])

                self.assertEqual(cm.output, ['INFO:gumps.apps.parametric_sweep_timeseries:No new data to save'])

    def test_update_hdf5_exception(self):
        "test that update hdf5 raises an exception if the directory is not set"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
            processing_function=get_total,
            batch=batch)

        with self.assertRaises(RuntimeError):
            app.update_hdf5(factors=pd.DataFrame(), responses=[pd.DataFrame()])

    def test_update_hdf5(self):
        "write to a temporary directory and check that data can be written correctly"
        parameters = gumps.solvers.sampler.SamplerSolverParameters(
            number_of_samples = 8,
            lower_bound = {'a':1, 'b':2, 'c':3, 'd':4},
            upper_bound = {'a':5, 'b':6, 'c':7, 'd':8},
            sampler = "sobol"
            )

        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            app = gumps.apps.parametric_sweep_timeseries.ParametricSweepTimeSeriesApp(parameters=parameters,
                processing_function=get_total,
                batch=batch,
                directory=directory)

            factors = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6]})

            responses = [pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]}),
                        pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]}),
                        pd.DataFrame({'time':[0,1,2], 'x':[1,2,3], 'y':[4,5,6]})]
            responses = [x.set_index("time") for x in responses]


            app.update_hdf5(factors, responses)

            self.assertTrue((directory / "data.h5").exists())
            data = H5( (directory / "data.h5").as_posix() )
            data.load()
            self.assertEqual(data.root.factors.shape, (3,2))
            self.assertEqual(data.root.responses.shape, (3,3,2))
            self.assertEqual([i.decode() for i in data.root.factor_names], ['x', 'y'])
            self.assertEqual([i.decode() for i in data.root.response_names], ['x', 'y'])
            self.assertEqual(data.root.time_values.shape, (3,))

            app.update_hdf5(factors, responses)

            self.assertTrue((directory / "data.h5").exists())
            data = H5( (directory / "data.h5").as_posix() )
            data.load()
            self.assertEqual(data.root.factors.shape, (6,2))
            self.assertEqual(data.root.responses.shape, (6,3,2))
            self.assertEqual([i.decode() for i in data.root.factor_names], ['x', 'y'])
            self.assertEqual([i.decode() for i in data.root.response_names], ['x', 'y'])
            self.assertEqual(data.root.time_values.shape, (3,))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParametricSweepTimeSeries)
    unittest.TextTestRunner(verbosity=2).run(suite)
