# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pandas as pd
import xarray as xr

from gumps.kernels.time_kernel import TimeKernelExample
from gumps.solvers.iterative_solver import IterativeSolver, IterativeSolverParameters
from gumps.studies.batch_time_study import BatchTimeStudyMultiProcess, batch_runner
from gumps.studies.study import SimulationStudy
from gumps.common.parallel import Parallel
from gumps.studies.batch_time_study_example import BatchTimeSeriesExample

def trivial_processing_function(data:pd.DataFrame|None) -> pd.DataFrame|None:
    "convert the index to the time column"
    if data is None:
        return None
    return data.set_index("time")

def exception_handler(function, x, e):
    return None

def null_processor(data:pd.DataFrame|None) -> pd.DataFrame|None:
    "convert the index to the time column"
    return data

class TestBatchTimeStudyMultiProcess(unittest.TestCase):
    "test the new study interface"

    def setUp(self):
        "setup the test"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        self.batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel)

    def test_init(self):
        "test the init method"
        self.assertIsInstance(self.batch.parallel, Parallel)
        self.assertIsInstance(self.batch.study, SimulationStudy)

    def test_run(self):
        "test the run method along with the start and stop methods"
        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},])

        with self.batch:
            results = self.batch.run(input_data, processing_function=trivial_processing_function)

        self.assertEqual(results.ta.shape, (3, 100))
        self.assertEqual(results.tb.shape, (3, 100))
        self.assertEqual(results.tc.shape, (3, 100))
        self.assertEqual(results.td.shape, (3, 100))

    def test_full_results(self):
        "test the full results method"
        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},
            {'a':3, 'b':4, 'c':5},])

        with self.batch:
            self.batch.run(input_data, processing_function=trivial_processing_function)

        input_data_copy, results = self.batch.full_results()

        self.assertEqual(results.ta.shape, (4, 100))
        self.assertEqual(results.tb.shape, (4, 100))
        self.assertEqual(results.tc.shape, (4, 100))
        self.assertEqual(results.td.shape, (4, 100))
        pd.testing.assert_frame_equal(input_data, input_data_copy)

    def test_full_results_exception(self):
        "test the full results method"
        with self.assertRaises(Exception):
            self.batch.full_results()

class TestBatchTimeStudyMultiProcessException(unittest.TestCase):
    "test the batch time study with exceptions"

    def test_run_exception(self):
        "testing running with an exception"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=-10, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},])

        with self.assertRaises(ValueError):
            with batch:
                batch.run(input_data, processing_function=trivial_processing_function)

    def test_run_exception_handler(self):
        "testing running with an exception"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},
            {'a':-1, 'b':-2, 'c':-3},])

        input_data_correct = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},])

        with batch:
            batch.run(input_data, processing_function=trivial_processing_function)

            input_data_copy, results = batch.full_results()

        self.assertEqual(results.ta.shape, (3, 100))
        self.assertEqual(results.tb.shape, (3, 100))
        self.assertEqual(results.tc.shape, (3, 100))
        self.assertEqual(results.td.shape, (3, 100))
        pd.testing.assert_frame_equal(input_data_correct, input_data_copy)


    def test_run_exception_handler_all_fail(self):
        "testing running with an exception"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        input_data = pd.DataFrame([
            {'a':-1, 'b':2, 'c':3},
            {'a':-4, 'b':5, 'c':6},
            {'a':-2, 'b':3, 'c':4},
            {'a':-1, 'b':-2, 'c':-3},])

        with batch:
            batch.run(input_data, processing_function=trivial_processing_function)

            input_data_copy, results = batch.full_results()


        self.assertTrue(input_data_copy.empty)
        self.assertIsInstance(results, xr.Dataset)


    def test_parallel_run_log_progress(self):
        "test parallel running with progress logging"
        model_variables = {'names':['a', 'b', 'c', 'd']}
        problem = {'a' : 1,'b' : 2 , 'c' : 3, 'd' : 4, 'time':0}
        solver_settings = IterativeSolverParameters(time_start=0, time_end=10, time_points=100)

        kernel = TimeKernelExample(model_variables=model_variables)
        solver = IterativeSolver(problem=problem, solver_settings=solver_settings)

        study = SimulationStudy(kernel=kernel, solver=solver)

        parallel = Parallel(poolsize=4)

        batch = BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler,
                                           progress_logging=True, progress_interval=1)

        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},
            {'a':-1, 'b':-2, 'c':-3},])

        with self.assertLogs('gumps.studies.batch_time_study', level='DEBUG') as cm:
            with batch:
                batch.run(input_data, processing_function=trivial_processing_function)

            self.assertEqual(cm.output, ['INFO:gumps.studies.batch_time_study:Batch study progress: 1/4 = 25.00%',
                'INFO:gumps.studies.batch_time_study:Batch study progress: 2/4 = 50.00%',
                'INFO:gumps.studies.batch_time_study:Batch study progress: 3/4 = 75.00%',
                'INFO:gumps.studies.batch_time_study:Batch study progress: 4/4 = 100.00%'])

    def test_iter_run(self):
        "just make sure iter_run does not raise an exception"
        batch = BatchTimeSeriesExample()
        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},])

        batch.run(input_data, processing_function = null_processor)

        output_data = batch.iter_run(input_data = batch.input_data, processing_function = null_processor)

        for idx, _, data in output_data:
            pd.testing.assert_frame_equal(data, batch.results[idx], atol=1e-2, rtol=1e-2)

class TestBatchTimeRunner(unittest.TestCase):

    def test_batch_runner(self):
        data = {'x': 1}

        def run_function(data:dict) -> pd.DataFrame:
            return pd.DataFrame({'y': [data['x']]})

        def processing_function(frame:pd.DataFrame|None) -> pd.DataFrame|None:
            if frame is None:
                return None
            return frame

        def exception_handler(function, x:dict, e:Exception) -> Exception|None:
            if isinstance(e, ValueError):
                return None
            else:
                return e

        data_new, result = batch_runner(data, run_function, processing_function, exception_handler)

        self.assertEqual(data_new, data)
        pd.testing.assert_frame_equal(result, pd.DataFrame({'y': [1]}))

    def test_batch_runner_exception_handled(self):
        data = {'x': -1}

        def run_function(data:dict) -> pd.DataFrame:
            raise ValueError("negative x")

        def processing_function(frame:pd.DataFrame|None) -> pd.DataFrame|None:
            if frame is None:
                return None
            return frame

        def exception_handler(function, x:dict, e:Exception) -> Exception|None:
            if isinstance(e, ValueError):
                return None
            else:
                return e

        data_new, result = batch_runner(data, run_function, processing_function, exception_handler)

        self.assertEqual(data_new, data)
        self.assertIsNone(result)

    def test_batch_runner_exception_unhandled(self):
        data = {'x': -1}

        def run_function(data:dict) -> pd.DataFrame:
            raise RuntimeError("negative x")

        def processing_function(frame:pd.DataFrame|None) -> pd.DataFrame|None:
            if frame is None:
                return None
            return frame

        def exception_handler(function, x:dict, e:Exception) -> Exception|None:
            if isinstance(e, ValueError):
                return None
            else:
                return e

        with self.assertRaises(RuntimeError):
            batch_runner(data, run_function, processing_function, exception_handler)

    def test_batch_runner_no_exception_handler(self):
        data = {'x': -1}

        def run_function(data:dict) -> pd.DataFrame:
            raise RuntimeError("negative x")

        def processing_function(frame:pd.DataFrame|None) -> pd.DataFrame|None:
            if frame is None:
                return None
            return frame

        with self.assertRaises(RuntimeError):
            batch_runner(data, run_function, processing_function, None)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchTimeStudyMultiProcess)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchTimeStudyMultiProcessException)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchTimeRunner)
    unittest.TextTestRunner(verbosity=2).run(suite)
