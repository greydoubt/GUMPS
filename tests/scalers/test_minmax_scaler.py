# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the minmax scaler regressor"

import gumps.scalers.minmax_scaler
import sklearn.preprocessing
import pandas as pd

import unittest

class TestMinMaxScaler(unittest.TestCase):
    "test the minmax scaler"

    def test_initialize(self):
        "test the initialization"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        self.assertIsInstance(scaler.scaler, sklearn.preprocessing.MinMaxScaler)
        self.assertIsInstance(scaler.columns, list)

    def test_fitted_false(self):
        "test the fitted method"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        self.assertFalse(scaler.fitted)

    def test_fitted_true(self):
        "test the fitted method when fitted"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertTrue(scaler.fitted)

    def test_fit(self):
        "test the fit method"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.columns, ["x"])

    def test_fit_bounds(self):
        "test the fit method with bounds"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        lower_bound = pd.Series({'x':0})
        upper_bound = pd.Series({'x':10})
        scaler.fit(data, lower_bound=lower_bound, upper_bound=upper_bound)
        self.assertEqual(scaler.columns, ["x"])

    def test_transform(self):
        "test the transform method"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        transformed_data = scaler.transform(data)
        self.assertEqual(transformed_data.shape, (3, 1))
        self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_get_feature_names_out(self):
        "test get feature names out"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_feature_names_out(), ["x"])

    def test_get_params(self):
        "test get params"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {'clip': False, 'copy': True, 'feature_range': (0, 1)})

    def test_inverse_transform(self):
        "test the inverse_transform method"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.fit(data)
        transformed_data = scaler.transform(data)
        inverse_transformed_data = scaler.inverse_transform(transformed_data)
        self.assertEqual(inverse_transformed_data.shape, (3, 1))

    def test_fit_transform(self):
        "test the fit transform method"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        transformed_data = scaler.fit_transform(data)
        self.assertEqual(transformed_data.shape, (3, 1))

    def test_fit_transform_bounds(self):
        "test the fit transform method with bounds"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        lower_bound = pd.Series({'x':0})
        upper_bound = pd.Series({'x':10})
        transformed_data = scaler.fit_transform(data, lower_bound=lower_bound, upper_bound=upper_bound)
        self.assertEqual(transformed_data.shape, (3, 1))

    def test_partial_fit(self):
        "test the partial fit method"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.partial_fit(data)
        self.assertEqual(scaler.columns, ["x"])

    def test_set_params(self):
        "test the set params method"
        scaler = gumps.scalers.minmax_scaler.MinMaxScaler()
        data = pd.DataFrame({"x": [1, 2, 3]})
        scaler.set_params(clip=True)
        scaler.fit(data)
        self.assertEqual(scaler.get_params(), {'clip': True, 'copy': True, 'feature_range': (0, 1)})



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMinMaxScaler)
    unittest.TextTestRunner(verbosity=2).run(suite)