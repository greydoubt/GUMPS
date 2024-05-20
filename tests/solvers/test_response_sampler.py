# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#Create tests for the response sampler

import unittest
import itertools
import pandas as pd
from unittest.mock import MagicMock

from gumps.solvers.response_sampler import ResponseSampler, ResponseSamplerParameters, check_response
from gumps.studies.batch_sphere_study import BatchLineStudy


class TestCheckResponse(unittest.TestCase):
    "check the check_response decorator"
    def setUp(self):
        self.obj = MagicMock()
        self.obj.response = 'response'
        self.obj.request = 'request'

    def test_check_response_with_response_and_request(self):
        @check_response
        def func(obj):
            return True

        result = func(self.obj)
        self.assertTrue(result)

    def test_check_response_with_response_none(self):
        self.obj.response = None

        @check_response
        def func(obj):
            return True

        with self.assertRaises(RuntimeError):
            func(self.obj)

    def test_check_response_with_request_none(self):
        self.obj.request = None

        @check_response
        def func(obj):
            return True

        with self.assertRaises(RuntimeError):
            func(self.obj)


class TestResponseSampler(unittest.TestCase):


    def setUp(self):
        "initialize the ResponseSamplerParameters"
        self.solver_settings = ResponseSamplerParameters(
            baseline={'x1': 1.0, 'x2': 2.0, 'x3': 3.0},
            lower_bound={'x1': 0.0, 'x2': 1.0, 'x3': 2.0},
            upper_bound={'x1': 2.0, 'x2': 3.0, 'x3': 4.0},
            points_1d=5
        )
        self.sampler = ResponseSampler(solver_settings=self.solver_settings)


    def test_has_next_true(self):
        "test has_next"
        self.assertTrue(self.sampler.has_next())


    def test_has_next_false(self):
        "test that has_next is false when a response is present"
        self.sampler.response = pd.DataFrame()
        self.assertFalse(self.sampler.has_next())


    def test_generate_1d(self):
        "test generate_1d returns a dataframe with the correct number of points"
        result = self.sampler.generate_1d()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), self.solver_settings.points_1d * len(self.solver_settings.baseline))


    def test_generate_1d_keys(self):
        "test generate_1d_keys returns a dataframe with the correct number of points"
        result = self.sampler.generate_1d_keys('x1')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), self.solver_settings.points_1d)
        self.assertEqual(result['x1'].iloc[0], self.solver_settings.lower_bound['x1'])
        self.assertEqual(result['x1'].iloc[-1], self.solver_settings.upper_bound['x1'])
        self.assertEqual(result['x2'].iloc[0], self.solver_settings.baseline['x2'])
        self.assertEqual(result['x2'].iloc[-1], self.solver_settings.baseline['x2'])
        self.assertEqual(result['x3'].iloc[0], self.solver_settings.baseline['x3'])
        self.assertEqual(result['x3'].iloc[-1], self.solver_settings.baseline['x3'])


    def test_generate_2d(self):
        "test generate_2d returns a dataframe with the correct number of points"
        self.sampler.solver_settings.points_2d_per_dimension = 5
        result = self.sampler.generate_2d()
        self.assertIsInstance(result, pd.DataFrame)
        # 3 combinations AB, AC, BC with 25 points each
        self.assertEqual(len(result), self.solver_settings.points_2d_per_dimension ** 2 * len(self.solver_settings.baseline))


    def test_generate_2d_keys(self):
        self.sampler.solver_settings.points_2d_per_dimension = 5
        result = self.sampler.generate_2d_keys('x1', 'x2')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), self.solver_settings.points_2d_per_dimension ** 2)
        self.assertEqual(result['x1'].iloc[0], self.solver_settings.lower_bound['x1'])
        self.assertEqual(result['x1'].iloc[-1], self.solver_settings.upper_bound['x1'])
        self.assertEqual(result['x2'].iloc[0], self.solver_settings.lower_bound['x2'])
        self.assertEqual(result['x2'].iloc[-1], self.solver_settings.upper_bound['x2'])
        self.assertEqual(result['x3'].iloc[0], self.solver_settings.baseline['x3'])
        self.assertEqual(result['x3'].iloc[-1], self.solver_settings.baseline['x3'])


    def test_ask_with_points_1d(self):
        self.solver_settings.points_2d_per_dimension = None
        self.sampler.solver_settings = self.solver_settings
        request = self.sampler.ask()
        self.assertIsInstance(request, pd.DataFrame)

        for key in self.solver_settings.baseline.keys():
            keys_used = request.columns.difference([key])

            count = request[keys_used].eq(self.solver_settings.baseline[keys_used]).all(axis=1).sum()

            self.assertEqual(count, self.solver_settings.points_1d)

        pd.testing.assert_series_equal(request.iloc[0], self.solver_settings.baseline, check_names=False)


    def test_ask_with_points_2d_per_dimension(self):
        self.sampler.solver_settings.points_1d = None
        self.sampler.solver_settings.points_2d_per_dimension = 5
        request = self.sampler.ask()
        self.assertIsInstance(request, pd.DataFrame)

        baseline = self.solver_settings.baseline

        for key1, key2 in itertools.combinations(baseline.keys(), 2):
            #find the indexes where all values except the key is equal to the mean value in the input_df

            keys_used = request.columns.difference([key1, key2])

            count = request[keys_used].eq(baseline[keys_used]).all(axis=1).sum()

            self.assertEqual(count, self.solver_settings.points_2d_per_dimension ** 2)

        pd.testing.assert_series_equal(request.iloc[0], self.solver_settings.baseline, check_names=False)


    def test_ask_with_points_1d_and_points_2d_per_dimension(self):
        self.sampler.solver_settings.points_1d = 5
        self.sampler.solver_settings.points_2d_per_dimension = 5
        request = self.sampler.ask()
        self.assertIsInstance(request, pd.DataFrame)

        baseline = self.solver_settings.baseline

        for key in self.solver_settings.baseline.keys():
            keys_used = request.columns.difference([key])

            count = request[keys_used].eq(self.solver_settings.baseline[keys_used]).all(axis=1).sum()

            self.assertEqual(count, self.solver_settings.points_1d)

        for key1, key2 in itertools.combinations(baseline.keys(), 2):
            #find the indexes where all values except the key is equal to the mean value in the input_df

            keys_used = request.columns.difference([key1, key2])

            count = request[keys_used].eq(baseline[keys_used]).all(axis=1).sum()

            self.assertEqual(count, self.solver_settings.points_2d_per_dimension ** 2)

        pd.testing.assert_series_equal(request.iloc[0], self.solver_settings.baseline, check_names=False)


    def test_ask_with_no_points_set(self):
        self.sampler.solver_settings.points_1d = None
        self.sampler.solver_settings.points_2d_per_dimension = None
        with self.assertRaises(ValueError):
            self.sampler.ask()


    def test_tell(self):
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})
        self.sampler.request = df
        self.sampler.tell(df)
        pd.testing.assert_frame_equal(self.sampler.response, df)

    def test_tell_exception_call_before_ask(self):
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})
        with self.assertRaises(RuntimeError):
            self.sampler.tell(df)

    def test_tell_exception_mismatch_rows(self):
        df = pd.DataFrame({'x1': [1, 2], 'x2': [4, 5], 'x3': [7, 8]})
        self.sampler.request = df
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
        with self.assertRaises(ValueError):
            self.sampler.tell(df)

    def test_tell_update_request(self):
        df = pd.DataFrame({'x1': [1, 2], 'x2': [4, 5], 'x3': [7, 8]})
        self.sampler.request = df

        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})
        self.sampler.tell(df, df)
        pd.testing.assert_frame_equal(self.sampler.request, df)
        pd.testing.assert_frame_equal(self.sampler.response, df)


    def test_get_response(self):
        df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})
        self.sampler.response = df
        result = self.sampler.get_response()
        pd.testing.assert_frame_equal(result, df)


    def test_get_response_exception(self):
        with self.assertRaises(Exception):
            self.sampler.get_response()


    def test_split_response_with_no_response_set(self):
        self.sampler.request = pd.DataFrame({'x1': [1, 2, 3], 'x2': [2, 2, 3], 'x3': [3, 3, 3]})
        with self.assertRaises(Exception):
            self.sampler.split_response()


    def test_get_baseline_with_no_response_set(self):
        self.sampler.response = None
        with self.assertRaises(RuntimeError):
            self.sampler.get_baseline_response()


    def test_get_baseline_with_no_request_set(self):
        self.sampler.request = None
        with self.assertRaises(RuntimeError):
            self.sampler.get_baseline_response()


    def test_get_response_2d_with_no_response_set(self):
        self.sampler.response = None
        with self.assertRaises(RuntimeError):
            self.sampler.get_response_2d()


    def test_get_response_2d_with_no_request_set(self):
        self.sampler.request = None
        with self.assertRaises(RuntimeError):
            self.sampler.get_response_2d()


class ResponseSamplerBatch(unittest.TestCase):
    "test all the methods that need a batch solver to run first"

    def setUp(self):
        "initialize and run the batch solver"
        self.solver_settings = ResponseSamplerParameters(
            baseline={'x1': 1, 'x2': 2, 'x3': 3},
            lower_bound={'x1': 0, 'x2': 1, 'x3': 2},
            upper_bound={'x1': 2, 'x2': 3, 'x3': 4},
            points_1d=5,
            points_2d_per_dimension = 5
        )
        self.sampler = ResponseSampler(solver_settings=self.solver_settings)

        batch = BatchLineStudy(dict(self.solver_settings.baseline))

        points = self.sampler.ask()

        response = batch.run(points, lambda x: x)

        self.sampler.tell(response)


    def test_split_response_with_response_set(self):
        "test split_response with response set"
        baseline, data_1d, data_2d = self.sampler.split_response()
        self.assertIsInstance(baseline, pd.DataFrame)
        self.assertIsInstance(data_1d, dict)
        self.assertIsInstance(data_2d, dict)


    def test_get_response_1d(self):
        data_1d = self.sampler.get_response_1d()
        self.assertIsInstance(data_1d, dict)
        self.assertIsInstance(data_1d['x1'], pd.DataFrame)
        self.assertIsInstance(data_1d['x2'], pd.DataFrame)
        self.assertIsInstance(data_1d['x3'], pd.DataFrame)


    def test_get_baseline(self):
        baseline = self.sampler.get_baseline_response()
        self.assertIsInstance(baseline, pd.DataFrame)

        correct = pd.Series({'x1': 1.0, 'x2': 2.0, 'x3': 3.0,
                                'd1': 0.0, 'd2': 0.0, 'd3': 0.0,
                                'total': 0.0}).to_frame().T
        pd.testing.assert_frame_equal(baseline, correct)


    def test_get_response_2d(self):
        response_2d = self.sampler.get_response_2d()
        self.assertIsInstance(response_2d, dict)
        self.assertEqual(len(response_2d), 3)
        self.assertIn('x1___x2', response_2d)
        self.assertIn('x1___x3', response_2d)
        self.assertIn('x2___x3', response_2d)
        self.assertIsInstance(response_2d['x1___x2'], pd.DataFrame)
        self.assertIsInstance(response_2d['x1___x3'], pd.DataFrame)
        self.assertIsInstance(response_2d['x2___x3'], pd.DataFrame)
        self.assertEqual(response_2d['x1___x2'].shape, (25, 6))
        self.assertEqual(response_2d['x1___x3'].shape, (25, 6))
        self.assertEqual(response_2d['x2___x3'].shape, (25, 6))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResponseSampler)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestCheckResponse)
    unittest.TextTestRunner(verbosity=2).run(suite)
