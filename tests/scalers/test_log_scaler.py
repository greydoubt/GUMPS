# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the minmax scaler regressor"

import gumps.scalers.log_scaler
import pandas as pd
import numpy as np

import unittest

class TestLogScaler(unittest.TestCase):
    "test the minmax scaler"

    def test_initialize(self):
        "test the initialization"
        scaler = gumps.scalers.log_scaler.LogScaler()
        self.assertIsInstance(scaler.log_columns, list)
        self.assertIsInstance(scaler.columns, list)

    def test_fit(self):
        "test the fit method"
        scaler = gumps.scalers.log_scaler.LogScaler()

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 6.0],
                             'z': [1000.0, 2.0, 0.0]})
        scaler.fit(data)
        self.assertEqual(scaler.columns, ["x", "y", "z"])
        self.assertEqual(scaler.log_columns, ["y"])

    def test_transform(self):
        "test the transform method"
        scaler = gumps.scalers.log_scaler.LogScaler()

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 6.0],
                             'z': [1000.0, 2.0, 0.0]})

        correct  = data.copy()
        correct['y'] = np.log(correct['y'])

        scaler.fit(data)
        transformed_data = scaler.transform(data)
        self.assertEqual(transformed_data.shape, (3, 3))
        self.assertIsInstance(transformed_data, pd.DataFrame)
        pd.testing.assert_frame_equal(transformed_data, correct)

    def test_get_feature_names_out(self):
        "test get feature names out"
        scaler = gumps.scalers.log_scaler.LogScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_feature_names_out(), ["x"])

    def test_get_params(self):
        "test get params"
        scaler = gumps.scalers.log_scaler.LogScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {})

    def test_inverse_transform(self):
        "test the inverse_transform method"
        scaler = gumps.scalers.log_scaler.LogScaler()

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 6.0],
                             'z': [1000.0, 2.0, 0.0]})

        inverse_data  = data.copy()
        inverse_data['y'] = np.log(inverse_data['y'])

        scaler.fit(data)
        inverse_transformed_data = scaler.inverse_transform(inverse_data)
        self.assertEqual(inverse_transformed_data.shape, (3, 3))
        self.assertIsInstance(inverse_transformed_data, pd.DataFrame)
        pd.testing.assert_frame_equal(inverse_transformed_data, data)

    def test_fit_transform(self):
        "test the fit transform method"
        scaler = gumps.scalers.log_scaler.LogScaler()

        data = pd.DataFrame({"x": [1.0, 2.0, 3.0],
                             'y': [1000.0, 1.0, 6.0],
                             'z': [1000.0, 2.0, 0.0]})

        correct  = data.copy()
        correct['y'] = np.log(correct['y'])

        transformed_data = scaler.fit_transform(data)
        self.assertEqual(transformed_data.shape, (3, 3))
        self.assertIsInstance(transformed_data, pd.DataFrame)
        pd.testing.assert_frame_equal(transformed_data, correct)

    def test_partial_fit(self):
        "test the partial fit method raises an error"
        scaler = gumps.scalers.log_scaler.LogScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})

        with self.assertRaises(NotImplementedError):
            scaler.partial_fit(data)

    def test_set_params(self):
        "test the set params method, this method does nothing since there are no parameters"
        scaler = gumps.scalers.log_scaler.LogScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.set_params(a=None)
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {})

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLogScaler)
    unittest.TextTestRunner(verbosity=2).run(suite)