# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pandas as pd

from gumps.common.parallel import Parallel
from gumps.kernels.sphere_kernel import SphereKernel
from gumps.solvers.simple_solver import SimpleSolver
from gumps.studies.batch_sphere_study import BatchSphereStudy
from gumps.studies.batch_study import BatchStudyMultiProcess, batch_runner
from gumps.studies.study import SimulationStudy, TestStudy
from gumps.studies.batch_sphere_study import BatchLineStudy

def exception_handler(function, x, e):
    return None

def get_total(frame:pd.DataFrame|None) -> dict|None:
    if frame is None:
        return None
    return {'total': frame.total[0]}

def get_total_exception(frame:pd.DataFrame|None) -> dict|None:
    if frame is None:
        return None
    return {'y': frame.y[0]}

def null_processor(frame:pd.DataFrame|None) -> pd.DataFrame|None:
    return frame

class TestBatchStudy(unittest.TestCase):
    "test the new study interface"

    def test_parallel_start_stop(self):
        "test start and stop interface"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_1':1.6, 'x_2': 0.7, 'x_3': 1.1, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        with self.assertLogs('gumps.common.parallel', level='INFO') as cm:
            batch.start()
        self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 4 workers'])

        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            batch.stop()
        self.assertEqual(cm.output, ['DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_contextmanager(self):
        "test start and stop interface"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_1':1.6, 'x_2': 0.7, 'x_3': 1.1, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        with self.assertLogs('gumps.common.parallel', level='DEBUG') as cm:
            with batch:
                pass
            self.assertEqual(cm.output, ['INFO:gumps.common.parallel:Using parallel pool with 4 workers',
                    'DEBUG:gumps.common.parallel:parallel pool closing down'])

    def test_parallel_run(self):
        "test parallel running"

        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_1':1.6, 'x_2': 0.7, 'x_3': 1.1, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([
            {'x_0': 1.1, 'x_1':1.2, 'x_2': 0.5, 'x_3': 1.3, 'n':4},
            {'x_0': 1.2, 'x_1':1.3, 'x_2': 0.6, 'x_3': 1.4, 'n':4},
            {'x_0': 1.3, 'x_1':1.5, 'x_2': 0.4, 'x_3': 1.5, 'n':4},
            {'x_0': 1.4, 'x_1':1.1, 'x_2': 0.3, 'x_3': 1.6, 'n':4}
        ])

        with batch:
            totals = batch.run(input_data, get_total)

        correct = pd.DataFrame({'total':[4.29, 5.07, 6.03, 5.74]})

        pd.testing.assert_frame_equal(totals, correct)


    def test_parallel_run_log_progress(self):
        "test parallel running with progress logging"
        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_1':1.6, 'x_2': 0.7, 'x_3': 1.1, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel, progress_logging=True, progress_interval=1)

        input_data = pd.DataFrame([
            {'x_0': 1.1, 'x_1':1.2, 'x_2': 0.5, 'x_3': 1.3, 'n':4},
            {'x_0': 1.2, 'x_1':1.3, 'x_2': 0.6, 'x_3': 1.4, 'n':4},
            {'x_0': 1.3, 'x_1':1.5, 'x_2': 0.4, 'x_3': 1.5, 'n':4},
            {'x_0': 1.4, 'x_1':1.1, 'x_2': 0.3, 'x_3': 1.6, 'n':4}
        ])

        with self.assertLogs('gumps.studies.batch_study', level='DEBUG') as cm:
            with batch:
                batch.run(input_data, get_total)

            self.assertEqual(cm.output, ['INFO:gumps.studies.batch_study:Batch study progress: 1/4 = 25.00%',
                'INFO:gumps.studies.batch_study:Batch study progress: 2/4 = 50.00%',
                'INFO:gumps.studies.batch_study:Batch study progress: 3/4 = 75.00%',
                'INFO:gumps.studies.batch_study:Batch study progress: 4/4 = 100.00%'])


    def test_sequential_run(self):
        "test parallel running"

        model_variables = {'n':4, 'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        problem = {'x_0': 1.5, 'x_1':1.6, 'x_2': 0.7, 'x_3': 1.1, 'n':4}
        solver = SimpleSolver(problem=problem, solver_settings=None)
        kernel = SphereKernel(model_variables=model_variables)

        study = SimulationStudy(solver, kernel)

        parallel = Parallel(poolsize=1)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([
            {'x_0': 1.1, 'x_1':1.2, 'x_2': 0.5, 'x_3': 1.3, 'n':4},
            {'x_0': 1.2, 'x_1':1.3, 'x_2': 0.6, 'x_3': 1.4, 'n':4},
            {'x_0': 1.3, 'x_1':1.5, 'x_2': 0.4, 'x_3': 1.5, 'n':4},
            {'x_0': 1.4, 'x_1':1.1, 'x_2': 0.3, 'x_3': 1.6, 'n':4}
        ])

        def get_total(frame:pd.DataFrame):
            return {'total': frame.total[0]}

        with batch:
            totals = batch.run(input_data, get_total)

        correct = pd.DataFrame({'total':[4.29, 5.07, 6.03, 5.74]})

        pd.testing.assert_frame_equal(totals, correct)

    def test_batch_sphere(self):
        "test parallel running"
        model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}

        batch = BatchSphereStudy(model_variables=model_variables)

        input_data = pd.DataFrame([
            {'x_0': 1.1, 'x_1':1.2, 'x_2': 0.5, 'x_3': 1.3},
            {'x_0': 1.2, 'x_1':1.3, 'x_2': 0.6, 'x_3': 1.4},
            {'x_0': 1.3, 'x_1':1.5, 'x_2': 0.4, 'x_3': 1.5},
            {'x_0': 1.4, 'x_1':1.1, 'x_2': 0.3, 'x_3': 1.6}
        ])

        def get_total(frame:pd.DataFrame):
            return pd.DataFrame(frame['total'])

        with batch:
            totals = batch.run(input_data, get_total)

        correct = pd.DataFrame({'total':[4.29, 5.07, 6.03, 5.74]})

        pd.testing.assert_frame_equal(totals, correct)

    def test_full_results(self):
        model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}

        batch = BatchSphereStudy(model_variables=model_variables)

        input_data_correct = pd.DataFrame([
            {'x_0': 1.1, 'x_1':1.2, 'x_2': 0.5, 'x_3': 1.3},
            {'x_0': 1.2, 'x_1':1.3, 'x_2': 0.6, 'x_3': 1.4},
            {'x_0': 1.3, 'x_1':1.5, 'x_2': 0.4, 'x_3': 1.5},
            {'x_0': 1.4, 'x_1':1.1, 'x_2': 0.3, 'x_3': 1.6}
        ])

        output_data_correct = pd.DataFrame([
            {'d_0': 1.00, 'd_1': 1.00, 'd_2': 0.04, 'd_3': 2.25, 'total': 4.29},
            {'d_0': 1.21, 'd_1': 1.21, 'd_2': 0.09, 'd_3': 2.56, 'total': 5.07},
            {'d_0': 1.44, 'd_1': 1.69, 'd_2': 0.01, 'd_3': 2.89, 'total': 6.03},
            {'d_0': 1.69, 'd_1': 0.81, 'd_2': 0.00, 'd_3': 3.24, 'total': 5.74}
        ])

        batch.input_data = input_data_correct
        batch.results = output_data_correct

        input_data, output_data = batch.full_results()

        pd.testing.assert_frame_equal(input_data_correct, input_data)
        pd.testing.assert_frame_equal(output_data_correct, output_data)


    def test_full_results_exception(self):
        model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}

        batch = BatchSphereStudy(model_variables=model_variables)

        with self.assertRaises(Exception):
            batch.full_results()

    def test_batch_sphere_results(self):
        "test parallel running"
        model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}

        batch = BatchSphereStudy(model_variables=model_variables)

        input_data = pd.DataFrame([
            {'x_0': 1.1, 'x_1':1.2, 'x_2': 0.5, 'x_3': 1.3},
            {'x_0': 1.2, 'x_1':1.3, 'x_2': 0.6, 'x_3': 1.4},
            {'x_0': 1.3, 'x_1':1.5, 'x_2': 0.4, 'x_3': 1.5},
            {'x_0': 1.4, 'x_1':1.1, 'x_2': 0.3, 'x_3': 1.6}
        ])

        def get_total(frame:pd.DataFrame):
            return pd.DataFrame(frame['total'])

        with batch:
            batch.run(input_data, get_total)

        original_input, results = batch.full_results()

        correct_results = pd.DataFrame({'d_0': [1.00, 1.21, 1.44, 1.69],
            'd_1': [1.00, 1.21, 1.69, 0.81],
            'd_2': [0.04, 0.09, 0.01, 0.00],
            'd_3': [2.25, 2.56, 2.89, 3.24],
            'total':[4.29, 5.07, 6.03, 5.74]})

        pd.testing.assert_frame_equal(original_input, input_data)
        pd.testing.assert_frame_equal(correct_results, results)


    def test_batch_study_multiprocess_exception(self):
        "test that using TestStudy with BatchStudyMultiProcess raises an exception with a negative x input"
        study = TestStudy(-5)

        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([
            {'x': 1},
            {'x': 2},
            {'x': 3},
            {'x': -1}
        ])

        def get_total(frame:pd.DataFrame|None) -> dict|None:
            if frame is None:
                return None
            return {'y': frame.y[0]}

        with self.assertRaises(Exception):
            with batch:
                batch.run(input_data, get_total)

    def test_batch_study_multiprocess_exception_handle(self):
        "test that using TestStudy with BatchStudyMultiProcess raises an exception with a negative x input"
        study = TestStudy(-5)

        parallel = Parallel(poolsize=4)

        batch = BatchStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        input_data = pd.DataFrame([
            {'x': 1},
            {'x': 2},
            {'x': 3},
            {'x': -1}
        ])

        with batch:
            batch.run(input_data, get_total_exception)

            input_data, output_data = batch.full_results()

            correct_input = pd.DataFrame([
                {'x': 1},
                {'x': 2},
                {'x': 3}
            ])

            correct_output = pd.DataFrame([
                {'y': 1},
                {'y': 2},
                {'y': 3}
            ])

            pd.testing.assert_frame_equal(input_data, correct_input)
            pd.testing.assert_frame_equal(output_data, correct_output)


    def test_iter_run(self):
        "just make sure iter_run does not raise an exception"
        model_variables = {'a_0': 0.1, 'a_1':0.2, 'a_2': 0.3, 'a_3': -0.2}
        batch = BatchLineStudy(model_variables=model_variables)

        input_data = pd.DataFrame([
            {'a_0': 1.1, 'a_1':1.2, 'a_2': 0.5, 'a_3': 1.3},
            {'a_0': 1.2, 'a_1':1.3, 'a_2': 0.6, 'a_3': 1.4},
            {'a_0': 1.3, 'a_1':1.5, 'a_2': 0.4, 'a_3': 1.5},
            {'a_0': 1.4, 'a_1':1.1, 'a_2': 0.3, 'a_3': 1.6}
        ])
        batch.run(input_data, processing_function = null_processor)

        output_data = batch.iter_run(input_data = batch.input_data, processing_function = null_processor)

        for idx, _, data in output_data:
            out = batch.results.iloc[idx]
            out.name = None
            pd.testing.assert_series_equal(data, out, atol=1e-2, rtol=1e-2)


class TestBatchRunner(unittest.TestCase):

    def test_batch_runner(self):
        data = {'x': 1}

        def run_function(data:dict) -> pd.DataFrame:
            return pd.DataFrame({'y': [data['x']]})

        def processing_function(frame:pd.DataFrame|None) -> dict|pd.Series|None:
            if frame is None:
                return None
            return {'y': frame.y[0]}

        def exception_handler(function, x:dict, e:Exception) -> Exception|None:
            if isinstance(e, ValueError):
                return None
            else:
                return e

        data_new, result = batch_runner(data, run_function, processing_function, exception_handler)

        self.assertEqual(data_new, data)
        self.assertEqual(result, {'y': 1})

    def test_batch_runner_exception_handled(self):
        data = {'x': -1}

        def run_function(data:dict) -> pd.DataFrame:
            raise ValueError("negative x")

        def processing_function(frame:pd.DataFrame|None) -> dict|pd.Series|None:
            if frame is None:
                return None
            return {'y': frame.y[0]}

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

        def processing_function(frame:pd.DataFrame|None) -> dict|pd.Series|None:
            if frame is None:
                return None
            return {'y': frame.y[0]}

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

        def processing_function(frame:pd.DataFrame|None) -> dict|pd.Series|None:
            if frame is None:
                return None
            return {'y': frame.y[0]}

        with self.assertRaises(RuntimeError):
            batch_runner(data, run_function, processing_function, None)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchStudy)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchRunner)
    unittest.TextTestRunner(verbosity=2).run(suite)