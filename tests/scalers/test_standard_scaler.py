# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the standard scaler regressor"

import gumps.scalers.standard_scaler
import sklearn.preprocessing
import pandas as pd

import unittest

class TestStandardScaler(unittest.TestCase):
    "test the standard scaler"

    def test_initialize(self):
        "test the initialization"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        self.assertIsInstance(scaler.scaler, sklearn.preprocessing.StandardScaler)
        self.assertIsInstance(scaler.columns, list)

    def test_fit(self):
        "test the fit method"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.columns, ["x"])

    def test_transform(self):
        "test the transform method"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        transformed_data = scaler.transform(data)
        self.assertEqual(transformed_data.shape, (3, 1))
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_get_feature_names_out(self):
        "test get feature names out"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_feature_names_out(), ["x"])

    def test_get_params(self):
        "test get params"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {'copy': True, 'with_mean': True, 'with_std': True})

    def test_inverse_transform(self):
        "test the inverse_transform method"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        transformed_data = scaler.transform(data)
        inverse_transformed_data = scaler.inverse_transform(transformed_data)
        self.assertEqual(inverse_transformed_data.shape, (3, 1))

    def test_fit_transform(self):
        "test the fit transform method"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        transformed_data = scaler.fit_transform(data)
        self.assertEqual(transformed_data.shape, (3, 1))

    def test_partial_fit(self):
        "test the partial fit method"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.partial_fit(data)
        self.assertEqual(scaler.columns, ["x"])

    def test_set_params(self):
        "test the set params method"
        scaler = gumps.scalers.standard_scaler.StandardScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.set_params(with_mean=False)
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {'copy': True, 'with_mean': False, 'with_std': True})


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStandardScaler)
    unittest.TextTestRunner(verbosity=2).run(suite)