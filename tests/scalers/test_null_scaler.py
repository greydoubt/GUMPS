# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the null scaler regressor"

import gumps.scalers.null_scaler
import pandas as pd

import unittest

class TestNullScaler(unittest.TestCase):
    "test the null scaler"

    def test_initialize(self):
        "test the initialization"
        scaler = gumps.scalers.null_scaler.NullScaler()
        self.assertIsNone(scaler.scaler)
        self.assertIsInstance(scaler.columns, list)

    def test_fit(self):
        "test the fit method"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.columns, ["x"])

    def test_transform(self):
        "test the transform method"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        transformed_data = scaler.transform(data)
        pd.testing.assert_frame_equal(transformed_data, data)

    def test_get_feature_names_out(self):
        "test get feature names out"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_feature_names_out(), ["x"])

    def test_get_params(self):
        "test get params"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {})

    def test_inverse_transform(self):
        "test the inverse_transform method"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        transformed_data = scaler.transform(data)
        inverse_transformed_data = scaler.inverse_transform(transformed_data)
        pd.testing.assert_frame_equal(inverse_transformed_data, data)

    def test_fit_transform(self):
        "test the fit transform method"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        transformed_data = scaler.fit_transform(data)
        pd.testing.assert_frame_equal(transformed_data, data)

    def test_partial_fit(self):
        "test the partial fit method"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.partial_fit(data)
        self.assertEqual(scaler.columns, ["x"])

    def test_set_params(self):
        "test the set params method"
        scaler = gumps.scalers.null_scaler.NullScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.set_params(clip=True)
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {})


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNullScaler)
    unittest.TextTestRunner(verbosity=2).run(suite)
