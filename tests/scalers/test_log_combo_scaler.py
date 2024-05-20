# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the minmax scaler regressor"

import unittest

import pandas as pd
import numpy as np

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.log_scaler import LogScaler
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.standard_scaler import StandardScaler


class TestLogComboScaler(unittest.TestCase):
    "test the minmax scaler"

    def test_initialize_standardscaler(self):
        "test the initialization"
        scaler = LogComboScaler(StandardScaler())
        self.assertIsInstance(scaler.log_scaler, LogScaler)
        self.assertIsInstance(scaler.scaler, StandardScaler)

    def test_initialize_minmaxscaler(self):
        "test the initialization with a minmax scaler"
        scaler = LogComboScaler(MinMaxScaler())
        self.assertIsInstance(scaler.log_scaler, LogScaler)
        self.assertIsInstance(scaler.scaler, MinMaxScaler)

    def test_fit(self):
        "test the fit method"
        scaler = LogComboScaler(StandardScaler())

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 6.0],
                             'z': [1000.0, 2.0, 0.0]})
        scaler.fit(data)
        self.assertEqual(scaler.log_scaler.columns, ["x", "y", "z"])
        self.assertEqual(scaler.log_scaler.log_columns, ["y"])

    def test_transform(self):
        "test the transform method"
        scaler = LogComboScaler(MinMaxScaler())

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 10.0],
                             'z': [1000.0, 500.0, 0.0]})

        correct = pd.DataFrame({"x": [0.0, 0.5, 1.0],
                             'y': [1.0, 0.0, 0.33333333333],
                             'z': [1.0, 0.5, 0.0]})

        scaler.fit(data)
        transformed_data = scaler.transform(data)
        self.assertEqual(transformed_data.shape, (3, 3))
        self.assertIsInstance(transformed_data, pd.DataFrame)
        pd.testing.assert_frame_equal(transformed_data, correct)

    def test_get_feature_names_out(self):
        "test get feature names out"
        scaler = LogComboScaler(MinMaxScaler())
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_feature_names_out(), ["x"])

    def test_get_params(self):
        "test get params"
        scaler = LogComboScaler(MinMaxScaler())
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {})

    def test_inverse_transform(self):
        "test the inverse_transform method"
        scaler = LogComboScaler(MinMaxScaler())

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 10.0],
                             'z': [1000.0, 500.0, 0.0]})

        inverse_data = pd.DataFrame({"x": [0.0, 0.5, 1.0],
                             'y': [1.0, 0.0, 0.33333333333],
                             'z': [1.0, 0.5, 0.0]})

        scaler.fit(data)
        inverse_transformed_data = scaler.inverse_transform(inverse_data)
        self.assertEqual(inverse_transformed_data.shape, (3, 3))
        self.assertIsInstance(inverse_transformed_data, pd.DataFrame)
        pd.testing.assert_frame_equal(inverse_transformed_data, data)

    def test_fit_transform(self):
        "test the fit transform method"
        scaler = LogComboScaler(MinMaxScaler())

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 10.0],
                             'z': [1000.0, 500.0, 0.0]})

        correct = pd.DataFrame({"x": [0.0, 0.5, 1.0],
                             'y': [1.0, 0.0, 0.33333333333],
                             'z': [1.0, 0.5, 0.0]})

        transformed_data = scaler.fit_transform(data)
        self.assertEqual(transformed_data.shape, (3, 3))
        self.assertIsInstance(transformed_data, pd.DataFrame)
        pd.testing.assert_frame_equal(transformed_data, correct)

    def test_partial_fit(self):
        "test the partial fit method raises an error"
        scaler = LogComboScaler(MinMaxScaler())
        data = pd.DataFrame({"x": [1, 2, 3]})

        with self.assertRaises(NotImplementedError):
            scaler.partial_fit(data)

    def test_set_params(self):
        "test the set params method, this method does nothing since there are no parameters"
        scaler = LogComboScaler(MinMaxScaler())
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.set_params(a=None)
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {})

    def test_get_bounds_transformed_none(self):
        scaler = LogComboScaler(MinMaxScaler())
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.set_params(a=None)
        scaler.fit(data)

        lower_bound, upper_bound = scaler.get_bounds_transformed(None, None)

        self.assertIsNone(lower_bound, upper_bound)

    def test_get_bounds_transformed(self):
        scaler = LogComboScaler(MinMaxScaler())
        data = pd.DataFrame({"x": np.exp([1, 2, 3])})
        scaler.set_params(a=None)

        lower_bound = pd.Series({'x': 1.0})
        upper_bound = pd.Series({'x': np.exp(15)})

        scaler.fit(data, lower_bound=lower_bound, upper_bound=upper_bound)

        lower_bound, upper_bound = scaler.get_bounds_transformed(lower_bound, upper_bound)

        self.assertIsInstance(lower_bound, pd.Series)
        self.assertIsInstance(upper_bound, pd.Series)
        pd.testing.assert_series_equal(lower_bound, pd.Series({'x': 0.0}))
        pd.testing.assert_series_equal(upper_bound, pd.Series({'x': 15.0}))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLogComboScaler)
    unittest.TextTestRunner(verbosity=2).run(suite)