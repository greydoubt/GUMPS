# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import pandas as pd
import numpy as np

from gumps.studies.batch_time_study_example import BatchTimeSeriesExample

def preprocessing_function(data:pd.DataFrame) -> pd.DataFrame:
    "make a test pre-processing function"
    min_data = data.min()
    mean_data = data.mean()
    max_data = data.max()

    min_data['name'] = 'min'
    mean_data['name'] = 'mean'
    max_data['name'] = 'max'

    temp = [min_data, mean_data, max_data]
    return pd.concat(temp, axis=1).T.set_index('name')

class TestBatchTimeSeriesExample(unittest.TestCase):
    "test the new study interface"

    def test_init(self):
        "test the init method"
        study = BatchTimeSeriesExample()
        self.assertEqual(study.model_variables, {'time_start':0, 'time_end':10, 'time_points':100})
        np.testing.assert_array_equal(study.time_series, np.linspace(0, 10, 100))

    def test_init_model_variables(self):
        "test the init method"
        study = BatchTimeSeriesExample(model_variables={'time_start':5, 'time_end':10, 'time_points':100})
        self.assertEqual(study.model_variables, {'time_start':5, 'time_end':10, 'time_points':100})

    def test_run_row(self):
        "test the run row method"
        study = BatchTimeSeriesExample()
        row = pd.Series({'a':1, 'b':2, 'c':3})
        result = study.run_row(row)
        self.assertEqual(result.shape, (100, 3))

    def test_run(self):
        "test the run method along with the start and stop methods"
        study = BatchTimeSeriesExample()
        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},])

        with study:
            results = study.run(input_data)

        self.assertEqual(results.ta.shape, (3, 100))
        self.assertEqual(results.tb.shape, (3, 100))
        self.assertEqual(results.tc.shape, (3, 100))

    def test_run_with_preprocessing(self):
        "test the run method along with the start and stop methods"
        study = BatchTimeSeriesExample()
        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},])

        with study:
            results = study.run(input_data, preprocessing_function)

        self.assertEqual(results.ta.shape, (3, 3))
        self.assertEqual(results.tb.shape, (3, 3))
        self.assertEqual(results.tc.shape, (3, 3))

    def test_full_results(self):
        "test the full results method"
        study = BatchTimeSeriesExample()
        input_data = pd.DataFrame([
            {'a':1, 'b':2, 'c':3},
            {'a':4, 'b':5, 'c':6},
            {'a':2, 'b':3, 'c':4},])

        with study:
            study.run(input_data)

        input_data_copy, results = study.full_results()

        self.assertEqual(results[0].shape, (100, 3))
        self.assertEqual(results[1].shape, (100, 3))
        self.assertEqual(results[2].shape, (100, 3))
        pd.testing.assert_frame_equal(input_data, input_data_copy)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchTimeSeriesExample)
    unittest.TextTestRunner(verbosity=2).run(suite)
