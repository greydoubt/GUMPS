# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import numpy as np
import pandas as pd

import gumps.loss.loss

target = pd.DataFrame([
    {'time':0, 'x1':1.0, 'x2': 0.5},
    {'time':1, 'x1':1.5, 'x2': 0.6},
    {'time':2, 'x1':2.0, 'x2': 0.7},
    {'time':3, 'x1':2.5, 'x2': 0.8},
    {'time':4, 'x1':3.0, 'x2': 0.9},
    {'time':5, 'x1':3.5, 'x2': 1.0}])

input_data = [
    pd.DataFrame([
        {'time':0, 'x1':1.1, 'x2': 0.57},
        {'time':1, 'x1':1.6, 'x2': 0.67},
        {'time':2, 'x1':2.1, 'x2': 0.77},
        {'time':3, 'x1':2.6, 'x2': 0.87},
        {'time':4, 'x1':3.1, 'x2': 0.97},
        {'time':5, 'x1':3.6, 'x2': 1.07}]),
    pd.DataFrame([
        {'time':0, 'x1':1.01, 'x2': 0.58},
        {'time':1, 'x1':1.51, 'x2': 0.68},
        {'time':2, 'x1':2.01, 'x2': 0.78},
        {'time':3, 'x1':2.51, 'x2': 0.88},
        {'time':4, 'x1':3.01, 'x2': 0.98},
        {'time':5, 'x1':3.51, 'x2': 1.08}]),
    pd.DataFrame([
        {'time':0, 'x1':1.05, 'x2': 0.51},
        {'time':1, 'x1':1.55, 'x2': 0.61},
        {'time':2, 'x1':2.05, 'x2': 0.71},
        {'time':3, 'x1':2.55, 'x2': 0.81},
        {'time':4, 'x1':3.05, 'x2': 0.91},
        {'time':5, 'x1':3.55, 'x2': 1.01}]),
    pd.DataFrame([
        {'time':0, 'x1':1.0, 'x2': 0.5},
        {'time':1, 'x1':1.5, 'x2': 0.6},
        {'time':2, 'x1':2.0, 'x2': 0.7},
        {'time':3, 'x1':2.5, 'x2': 0.8},
        {'time':4, 'x1':3.0, 'x2': 0.9},
        {'time':5, 'x1':3.5, 'x2': 1.0}])
    ]

class TestLoss(unittest.TestCase):
    "test the loss interface"

    def test_sse_batch(self):
        ""

        target_batch = pd.DataFrame([
            {'x1':0.06, 'x2':0.0294},
            {'x1':0.0006, 'x2':0.0384},
            {'x1':0.0150, 'x2':0.0006},
            {'x1':0.0, 'x2':0.0}
        ])

        input_data_batch = pd.DataFrame([
            {'x1':0.16, 'x2':0.1294},
            {'x1':0.1006, 'x2':0.1384},
            {'x1':0.1150, 'x2':0.1006},
            {'x1':1.0, 'x2':1.0}
        ])

        loss = gumps.loss.loss.SumSquaredErrorBatch(target=target_batch, weights=None)

        errors = loss.run(input_data_batch)

        correct = pd.DataFrame([
            {'x1':0.01, 'x2':0.01},
            {'x1':0.01, 'x2':0.01},
            {'x1':0.01, 'x2':0.01},
            {'x1':1.0, 'x2':1.0}
        ])

        pd.testing.assert_frame_equal(errors, correct)

    def test_sse_batch_dimensions(self):
        "test batch dimension mismatch"

        target_batch = pd.DataFrame([
            {'x1':0.06, 'x2':0.0294},
        ])

        input_data_batch = pd.DataFrame([
            {'x1':0.16, 'x2':0.1294, 'x3': 0.05},
        ])

        loss = gumps.loss.loss.SumSquaredErrorBatch(target=target_batch, weights=None)

        with self.assertRaises(ValueError):
            loss.check_dimensions(target_batch, input_data_batch)

    def test_sse_batch_weights(self):
        "test batch dimension mismatch"

        target_batch = pd.DataFrame([
            {'x1':0.06, 'x2':0.0294},
            {'x1':0.006, 'x2':0.00294},
        ])

        input_data_batch = pd.DataFrame([
            {'x1':0.16, 'x2':0.1294},
            {'x1':0.016, 'x2':0.01294},
        ])

        weights = {
				"Err1": {"x1":1, "x2":0},
				"Err2": {"x1":0, "x2":1},
				"Err3": {"x1":0.5, "x2":0.5}
			}

        loss = gumps.loss.loss.SumSquaredErrorBatch(target=target_batch, weights=weights)

        errors = loss.run(input_data_batch)

        correct = pd.DataFrame([
             {'Err1':0.01, 'Err2':0.01, 'Err3':0.01},
             {'Err1':0.0001, 'Err2':0.0001, 'Err3':0.0001}
        ])

        pd.testing.assert_frame_equal(errors, correct)


    def test_sse(self):
        ""
        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=None)

        errors = pd.DataFrame(loss.run(df) for df in input_data)

        correct = pd.DataFrame([
            {'x1':0.06, 'x2':0.0294},
            {'x1':0.0006, 'x2':0.0384},
            {'x1':0.0150, 'x2':0.0006},
            {'x1':0.0, 'x2':0.0}
        ])

        pd.testing.assert_frame_equal(errors, correct)

    def test_sse_weights(self):
        "test that weights distribute over multiple dataframes correctly"
        weights = {
				"Err1": {"x1":1, "x2":0},
				"Err2": {"x1":0, "x2":1},
				"Err3": {"x1":0.5, "x2":0.5}
			}

        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=weights)

        errors = pd.DataFrame(loss.run(df) for df in input_data)

        correct = pd.DataFrame([
            {'Err1':0.06, 'Err2':0.0294, 'Err3':0.0447},
            {'Err1':0.0006, 'Err2':0.0384, 'Err3':0.0195},
            {'Err1':0.0150, 'Err2':0.0006, 'Err3':0.0078},
            {'Err1':0.0, 'Err2':0.0, 'Err3':0}
        ])

        pd.testing.assert_frame_equal(errors, correct)

    def test_sse_weight(self):
        "test that weights can be applied to a single dataframe"
        weights = {
				"Err1": {"x1":1, "x2":0},
				"Err2": {"x1":0, "x2":1},
				"Err3": {"x1":0.5, "x2":0.5}
			}

        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=weights)

        errors = loss.calculate_loss(input_data[0])

        correct = pd.Series({'Err1':0.06, 'Err2':0.0294, 'Err3':0.0447})

        pd.testing.assert_series_equal(errors, correct)

    def test_sse_check_dimensions(self):
        "test that weights can be applied to a single dataframe"
        source_data = pd.DataFrame([
            {'time':0, 'x1':1.0, 'x2': 0.5},
            {'time':1, 'x1':1.5, 'x2': 0.6},
            {'time':2, 'x1':2.0, 'x2': 0.7},
            {'time':3, 'x1':2.5, 'x2': 0.8},
            {'time':4, 'x1':3.0, 'x2': 0.9},
            {'time':5, 'x1':3.5, 'x2': 1.0}])

        target_data = pd.DataFrame([
            {'time':0, 'x1':1.0, 'x2': 0.5},
            {'time':1, 'x1':1.5, 'x2': 0.6},
            {'time':2, 'x1':2.0, 'x2': 0.7}])

        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target_data, weights=None)

        with self.assertRaises(ValueError):
            loss.check_dimensions(source_data, target_data)

    def test_sse_get_data(self):
        "get the data"
        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=None)

        source = pd.DataFrame([
            {'time':0, 'x1':1.0, 'x2': 0.5},
            {'time':0.5, 'x1':1.0, 'x2': 0.5},
            {'time':1, 'x1':1.5, 'x2': 0.6},
            {'time':1.5, 'x1':1.5, 'x2': 0.6},
            {'time':2, 'x1':2.0, 'x2': 0.7},
            {'time':2.5, 'x1':2.0, 'x2': 0.7},
            {'time':3, 'x1':2.5, 'x2': 0.8},
            {'time':3.5, 'x1':2.5, 'x2': 0.8},
            {'time':4, 'x1':3.0, 'x2': 0.9},
            {'time':4.5, 'x1':3.0, 'x2': 0.9},
            {'time':5, 'x1':3.5, 'x2': 1.0}])

        subset_source, subset_target = loss.get_data(source)

        correct_source = pd.DataFrame([
            {'x1':1.0, 'x2': 0.5},
            {'x1':1.5, 'x2': 0.6},
            {'x1':2.0, 'x2': 0.7},
            {'x1':2.5, 'x2': 0.8},
            {'x1':3.0, 'x2': 0.9},
            {'x1':3.5, 'x2': 1.0}])

        correct_target = pd.DataFrame([
            {'x1':1.0, 'x2': 0.5},
            {'x1':1.5, 'x2': 0.6},
            {'x1':2.0, 'x2': 0.7},
            {'x1':2.5, 'x2': 0.8},
            {'x1':3.0, 'x2': 0.9},
            {'x1':3.5, 'x2': 1.0}])

        pd.testing.assert_frame_equal(subset_source, correct_source)
        pd.testing.assert_frame_equal(subset_target, correct_target)

    def test_sse_check_dimensions_pass(self):
        "check dimensions"
        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=None)

        test_df = pd.DataFrame([
            {'time':0, 'x1':1.1, 'x2': 0.57},
            {'time':1, 'x1':1.6, 'x2': 0.67},
            {'time':2, 'x1':2.1, 'x2': 0.77},
            {'time':3, 'x1':2.6, 'x2': 0.87},
            {'time':4, 'x1':3.1, 'x2': 0.97},
            {'time':5, 'x1':3.6, 'x2': 1.07}])

        source_data, target_data = loss.get_data(test_df)

        loss.check_dimensions(source_data ,target_data)


    def test_sse_check_dimensions_fail(self):
        "check that dimensions fail"
        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=None)

        test_df = pd.DataFrame([
            {'time':2, 'x1':2.1, 'x2': 0.77},
            {'time':3, 'x1':2.6, 'x2': 0.87},
            {'time':4, 'x1':3.1, 'x2': 0.97},
            {'time':5, 'x1':3.6, 'x2': 1.07}])

        source_data, target_data = loss.get_data(test_df)

        with self.assertRaises(ValueError):
            loss.check_dimensions(source_data ,target_data)

    def test_sse_check_dimensions_fail_log(self):
        "check that the right failure log is made"
        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=None)

        test_df = pd.DataFrame([
            {'time':2, 'x1':2.1, 'x2': 0.77},
            {'time':3, 'x1':2.6, 'x2': 0.87},
            {'time':4, 'x1':3.1, 'x2': 0.97},
            {'time':5, 'x1':3.6, 'x2': 1.07}])

        source_data, target_data = loss.get_data(test_df)

        with self.assertLogs('gumps.loss.loss', level='DEBUG') as cm:
            try:
                loss.check_dimensions(source_data ,target_data)
            except ValueError:
                pass
        self.assertEqual(cm.output, ["ERROR:gumps.loss.loss:source: (4, 2):['x1', 'x2']  target: (6, 2):['x1', 'x2']"])

    def test_apply_weights(self):
        "test apply weights"
        weights = {
				"Err1": {"x1":1, "x2":0},
				"Err2": {"x1":0, "x2":1},
				"Err3": {"x1":0.5, "x2":0.5}
			}

        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=weights)

        series = pd.Series({'x1':1, 'x2':0.5})

        weighted = loss.apply_weights(series)

        correct = pd.Series({'Err1':1, 'Err2':0.5, 'Err3':0.75})

        pd.testing.assert_series_equal(weighted, correct)

    def test_calculate_loss(self):
        "calculate the loss"
        loss = gumps.loss.loss.SumSquaredErrorTimeSeries(target=target, weights=None)

        test_df = pd.DataFrame([
            {'time':0, 'x1':1.1, 'x2': 0.57},
            {'time':1, 'x1':1.6, 'x2': 0.67},
            {'time':2, 'x1':2.1, 'x2': 0.77},
            {'time':3, 'x1':2.6, 'x2': 0.87},
            {'time':4, 'x1':3.1, 'x2': 0.97},
            {'time':5, 'x1':3.6, 'x2': 1.07}])

        error = loss.calculate_loss(test_df)

        correct = pd.Series({'x1':0.06, 'x2':0.0294})

        pd.testing.assert_series_equal(error, correct)

    def test_calculate_loss_nsse(self):
        "calculate the loss"
        loss = gumps.loss.loss.NormalizedSumSquaredErrorTimeSeries(target=target, weights=None)

        test_df = pd.DataFrame([
            {'time':0, 'x1':1.1, 'x2': 0.57},
            {'time':1, 'x1':1.6, 'x2': 0.67},
            {'time':2, 'x1':2.1, 'x2': 0.77},
            {'time':3, 'x1':2.6, 'x2': 0.87},
            {'time':4, 'x1':3.1, 'x2': 0.97},
            {'time':5, 'x1':3.6, 'x2': 1.07}])

        error = loss.calculate_loss(test_df)

        correct = pd.Series({'x1':0.00489796, 'x2':0.0294})

        pd.testing.assert_series_equal(error, correct)

    def test_calculate_loss_nrmse(self):
        "calculate the loss"
        loss = gumps.loss.loss.NormalizedRootMeanSquaredErrorTimeSeries(target=target, weights=None)

        test_df = pd.DataFrame([
            {'time':0, 'x1':1.1, 'x2': 0.57},
            {'time':1, 'x1':1.6, 'x2': 0.67},
            {'time':2, 'x1':2.1, 'x2': 0.77},
            {'time':3, 'x1':2.6, 'x2': 0.87},
            {'time':4, 'x1':3.1, 'x2': 0.97},
            {'time':5, 'x1':3.6, 'x2': 1.07}])

        error = loss.calculate_loss(test_df)

        correct = pd.Series({'x1':0.028571428, 'x2':0.07})

        pd.testing.assert_series_equal(error, correct)

    def test_calculate_loss_mase(self):
        "calculate the loss"
        loss = gumps.loss.loss.MeanAbsoluteStandardErrorTimeSeries(target=target, weights=None)

        test_df = pd.DataFrame([
            {'time':0, 'x1':1.1, 'x2': 0.57},
            {'time':1, 'x1':1.6, 'x2': 0.67},
            {'time':2, 'x1':2.1, 'x2': 0.77},
            {'time':3, 'x1':2.6, 'x2': 0.87},
            {'time':4, 'x1':3.1, 'x2': 0.97},
            {'time':5, 'x1':3.6, 'x2': 1.07}])

        error = loss.calculate_loss(test_df)

        correct = pd.Series({'x1':0.24, 'x2':0.84})

        pd.testing.assert_series_equal(error, correct)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLoss)
    unittest.TextTestRunner(verbosity=2).run(suite)
