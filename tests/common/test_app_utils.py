# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the app_utils module in common with python unitests"
import unittest
import gumps.studies
import gumps.common.parallel
import gumps.studies.batch_time_study
import gumps.studies.study
import gumps.studies.batch_study
from gumps.common.app_utils import run_batch_iterator, run_batch_time_iterator

import pandas as pd

def processing_function(frame: pd.DataFrame|None) -> pd.DataFrame|None:
    "process the dataframe for the loss function"
    if frame is None:
        return None
    return frame

def procession_function_single(data:pd.DataFrame) -> dict|pd.Series|None:
    "process the data for the loss function"
    if data is None:
        return None
    return pd.Series({'x': data['x'][0], 'y': data['y'][0]})

def exception_handler(function, x, e):
    return None

class TestAppUtils(unittest.TestCase):
    "test app_utils"

    def test_run_batch_time_iterator(self):
        "test run_batch_iterator"
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_time_study.BatchTimeStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([{'a':1, 'b':2, 'c':3}, {'a':4, 'b':5, 'c':6}])

        input_data_new, output_data_new = run_batch_time_iterator(batch, input_data, processing_function, None)

        self.assertEqual(input_data_new.shape, (2, 3))
        self.assertEqual(output_data_new.ta.shape, (2, 2))
        self.assertEqual(output_data_new.tb.shape, (2, 2))
        self.assertEqual(output_data_new.tc.shape, (2, 2))

    def test_run_batch_time_iterator_nan(self):
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_time_study.BatchTimeStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([{'a':1, 'b':2, 'c':3, 'nan':0.0}, {'a':4, 'b':5, 'c':6, 'nan':1.0}])

        input_data_new, output_data_new = run_batch_time_iterator(batch, input_data, processing_function, None)

        self.assertEqual(input_data_new.shape, (1, 4))
        self.assertEqual(output_data_new.ta.shape, (1, 2))
        self.assertEqual(output_data_new.tb.shape, (1, 2))
        self.assertEqual(output_data_new.tc.shape, (1, 2))

    def test_run_batch_time_iterator_fail(self):
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_time_study.BatchTimeStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        input_data = pd.DataFrame([{'a':1, 'b':2, 'c':3, 'fail':0.0}, {'a':4, 'b':5, 'c':6, 'fail':1.0}])

        input_data_new, output_data_new = run_batch_time_iterator(batch, input_data, processing_function, None)

        self.assertEqual(input_data_new.shape, (1, 4))
        self.assertEqual(output_data_new.ta.shape, (1, 2))
        self.assertEqual(output_data_new.tb.shape, (1, 2))
        self.assertEqual(output_data_new.tc.shape, (1, 2))

    def test_run_batch_time_iterator_pre_processing(self):
        "test run_batch_iterator"
        model_variables = {'time_start':0, 'time_end':1, 'time_points':2}
        problem = {'a':1, 'b':2, 'c':3}

        study = gumps.studies.TestStudyTime(model_variables=model_variables, problem=problem)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_time_study.BatchTimeStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([{'a':1, 'b':2, 'c':3}, {'a':4, 'b':5, 'c':6}])

        input_data_new, output_data_new = run_batch_time_iterator(batch, input_data, processing_function, lambda x: x)

        self.assertEqual(input_data_new.shape, (2, 3))
        self.assertEqual(output_data_new.ta.shape, (2, 2))
        self.assertEqual(output_data_new.tb.shape, (2, 2))
        self.assertEqual(output_data_new.tc.shape, (2, 2))


    def test_run_batch_iterator(self):
        "test run_batch_iterator"
        study = gumps.studies.TestStudy(5)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_study.BatchStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([{'x':1}, {'x':4}])

        input_data_new, output_data_new = run_batch_iterator(batch, input_data, procession_function_single, None)

        self.assertEqual(input_data_new.shape, (2, 1))
        self.assertEqual(output_data_new.shape, (2, 2))

    def test_run_batch_iterator_nan(self):
        study = gumps.studies.TestStudy(5)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_study.BatchStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([{'x':1}, {'x':11}])

        input_data_new, output_data_new = run_batch_iterator(batch, input_data, procession_function_single, None)

        self.assertEqual(input_data_new.shape, (1, 1))
        self.assertEqual(output_data_new.shape, (1, 2))

    def test_run_batch_iterator_fail(self):
        study = gumps.studies.TestStudy(-5)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_study.BatchStudyMultiProcess(study=study, parallel=parallel, exception_handler=exception_handler)

        input_data = pd.DataFrame([{'x':-1}, {'x':4}])

        input_data_new, output_data_new = run_batch_iterator(batch, input_data, procession_function_single, None)

        self.assertEqual(input_data_new.shape, (1, 1))
        self.assertEqual(output_data_new.shape, (1, 2))

    def test_run_batch_iterator_pre_processing(self):
        "test run_batch_iterator"
        study = gumps.studies.TestStudy(5)

        parallel = gumps.common.parallel.Parallel(poolsize=2)

        batch = gumps.studies.batch_study.BatchStudyMultiProcess(study=study, parallel=parallel)

        input_data = pd.DataFrame([{'x':1}, {'x':4}])

        input_data_new, output_data_new = run_batch_iterator(batch, input_data, procession_function_single, lambda x: x)

        self.assertEqual(input_data_new.shape, (2, 1))
        self.assertEqual(output_data_new.shape, (2, 2))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAppUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)