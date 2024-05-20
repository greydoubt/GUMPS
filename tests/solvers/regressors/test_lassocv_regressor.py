# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"lasso cross validation regressor test"

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures

from gumps.solvers.regressors.lassocv_regressor import (
    LassoCVRegressionParameters, LassoCVRegressor)
from gumps.solvers.regressors.regression_loader import load_regressor
from gumps.solvers.regressors.regression_solver import RegressionParameters
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.solvers.regressors.regressor_data import DataRegression


def generate_regression_parameters(count: int = 5):
    "generate regression parameters"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
    order = 3
    return LassoCVRegressionParameters(input_data=input_data, output_data=output_data, order=order)

def generate_regression_parameters_nd(count: int = 50):
    "generate regression parameters for more dimensions"
    input_data = pd.DataFrame({"x1": np.linspace(0, 1, count),
                               "x2": np.linspace(1, 2, count),
                               "x3": np.linspace(2, 3, count)})
    output_data = pd.DataFrame({"y1": np.linspace(0, 1, count),
                                "y2": np.linspace(1, 2, count)})
    order = 1
    return LassoCVRegressionParameters(input_data=input_data, output_data=output_data, order=order)

class TestLassoCVRegressionSolver(unittest.TestCase):
    "test the regressor"

    def test_polynomial_parameters_ok(self):
        "test the polynomial parameters"
        count = 5
        input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
        output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
        order = 2
        LassoCVRegressionParameters(input_data=input_data, output_data=output_data, order=order)

    def test_polynomial_parameters_bad(self):
        "test the polynomial parameters"
        count  = 5
        input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
        output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
        order = 4.5
        with self.assertRaises(ValueError):
            LassoCVRegressionParameters(input_data=input_data, output_data=output_data, order=order)

    def test_polynomial_parameters_exception_order(self):
        "test the polynomial parameters"
        count  = 5
        input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
        output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
        order = 5
        with self.assertRaises(ValueError):
            LassoCVRegressionParameters(input_data=input_data, output_data=output_data, order=order)

    def test_polynomial_parameters_logging(self):
        "test the polynomial logging"
        count  = 5
        input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
        output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
        order = 3
        with self.assertLogs('gumps.solvers.regressors.polynomial_regressor', level='INFO') as cm:
            LassoCVRegressionParameters(input_data=input_data, output_data=output_data, order=order)
        self.assertEqual(cm.output, ['INFO:gumps.solvers.regressors.polynomial_regressor:There are 5 samples and 4 polynomial terms.'])

    def test_initialize(self):
        "test the initialization"
        parameters = generate_regression_parameters()
        regressor = LassoCVRegressor(parameters)
        self.assertIsInstance(regressor.parameters, LassoCVRegressionParameters)
        self.assertIsInstance(regressor.data_regression.input_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.input_scaler, LogComboScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler, LogComboScaler)
        self.assertIsInstance(regressor.regressor, sklearn.linear_model.MultiTaskLassoCV)
        self.assertIsInstance(regressor.data_regression, DataRegression)
        self.assertIsInstance(regressor.poly, PolynomialFeatures)


    def test_get_scalers(self):
        "test the initialization of the input and output scalers"
        parameters = generate_regression_parameters()
        regressor = LassoCVRegressor(parameters)
        input_scaler, output_scaler = regressor._get_scalers()
        self.assertIsInstance(input_scaler, LogComboScaler)
        self.assertIsInstance(output_scaler, LogComboScaler)

    def test_get_regressor(self):
        "test the initialization of the regressor"
        parameters = generate_regression_parameters()
        regressor = LassoCVRegressor(parameters)
        reg = regressor._get_regressor()
        self.assertIsInstance(reg, sklearn.linear_model.MultiTaskLassoCV)

    def test_save(self):
        "test the save method"
        parameters = generate_regression_parameters(count=10)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            self.assertTrue((temp_dir / "regressor.joblib").exists())
            self.assertTrue((temp_dir / "parameters.joblib").exists())
            self.assertTrue((temp_dir / "data").exists())


    def test_load(self):
        "test the load method"
        parameters = generate_regression_parameters(count=10)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)
            new_regressor = LassoCVRegressor.load(temp_dir)

            self.assertEqual(str(regressor.regressor), str(new_regressor.regressor))

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.min_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.min_)

            pd.testing.assert_frame_equal(regressor.predict(parameters.input_data),
                                          new_regressor.predict(parameters.input_data))

    def test_generic_load(self):
        "test the load method"
        parameters = generate_regression_parameters(count=10)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)
            new_regressor = load_regressor(temp_dir)

            self.assertEqual(str(regressor.regressor), str(new_regressor.regressor))

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.min_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.min_)

            pd.testing.assert_frame_equal(regressor.predict(parameters.input_data),
                                          new_regressor.predict(parameters.input_data))


    def test_fit(self):
        "test the fit method"
        parameters = generate_regression_parameters(count=100)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

    def test_fit_nd(self):
        "test the fit nd method"
        parameters = generate_regression_parameters_nd(count=100)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

    def test_error_metrics(self):
        "test the error metrics method"
        parameters = generate_regression_parameters(count=100)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()
        pd.testing.assert_series_equal(pd.Series({'score': 1.0, 'r2_score': 1.0,
                                                  'mean_squared_error': 0.0, 'mse': 0.0,
                                                  'root_mean_squared_error': 0.0, 'rmse': 0.0,
                                                  'normalized_mean_squared_error': 0.0,'nmse': 0.0,
                                                  'normalized_root_mean_squared_error': 0.0, 'nrmse': 0.0}),
                                       regressor.error_metrics(), atol=1e-3, rtol=1e-3)

    def test_error_metrics_nd(self):
        "test the error metrics in nd"
        parameters = generate_regression_parameters_nd(count=100)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()
        pd.testing.assert_series_equal(pd.Series({'score': 1.0, 'r2_score': 1.0,
                                                  'mean_squared_error': 0.0, 'mse': 0.0,
                                                  'root_mean_squared_error': 0.0, 'rmse': 0.0,
                                                  'normalized_mean_squared_error': 0.0,'nmse': 0.0,
                                                  'normalized_root_mean_squared_error': 0.0, 'nrmse': 0.0}),
                                       regressor.error_metrics(), atol=1e-3, rtol=1e-3)

    def test_config(self):
        "test that we can set a configuration setting that is followed"
        parameters = generate_regression_parameters(count=100)
        parameters.max_iter = 1
        regressor = LassoCVRegressor(parameters)
        regressor.fit()
        self.assertTrue(regressor.regressor.n_iter_ == 1)


    def test_predict(self):
        "test the predict method"
        parameters = generate_regression_parameters(count=100)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

        input_data = pd.DataFrame({'x': np.linspace(0, 1, 5)})
        output_data_correct = pd.DataFrame({'y': np.linspace(0, 1, 5)})

        output_data = regressor.predict(input_data)
        pd.testing.assert_frame_equal(output_data_correct,
                                      output_data,
                                      atol=1e-2, rtol=1e-2)

    def test_predict_nd(self):
        "test the predict method in nd"
        parameters = generate_regression_parameters_nd(count=100)
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

        input_data = parameters.input_data
        output_data_correct = parameters.output_data

        output_data = regressor.predict(input_data)
        pd.testing.assert_frame_equal(output_data_correct,
                                      output_data,
                                      atol=1e-2, rtol=1e-2)


    def test_load_rebuild(self):
        "load the regressor with a different version of sklearn than it was saved with and auto-matically rebuild it"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "lassocv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.regressors.regression_solver', level='WARNING') as cm:
                load_regressor(temp_dir, auto_resave=False)
            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor failed to load, rebuilding regressor.' in cm.output)

    def test_load_rebuild_resave(self):
        "load the regressor with a different version of sklearn than it was saved with and auto-matically rebuild it"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "lassocv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.regressors.regression_solver', level='WARNING') as cm:
                load_regressor(temp_dir, auto_resave=True, auto_rebuild=True)
            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor failed to load, rebuilding regressor.' in cm.output)
            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor was rebuilt, resaving regressor.' in cm.output)

    def test_load_exception(self):
        "load the regressor with a different version of sklearn and disallow auto-rebuild"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "lassocv"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                load_regressor(temp_dir, auto_rebuild=False)

    def test_clone(self):
        "test the clone method"
        parameters = generate_regression_parameters()
        regressor = LassoCVRegressor(parameters)
        new_regressor = regressor.clone(parameters)
        self.assertEqual(regressor.parameters, new_regressor.parameters)

    def test_auto_tune(self):
        "test the auto tune method"
        parameters = generate_regression_parameters_nd()
        regressor = LassoCVRegressor(parameters)
        regressor.fit()

        regressor.auto_tune(max_order=4)
        after = regressor.get_tuned_parameters()

        correct = {'order': 4, 'terms': ['x1']}
        self.assertEqual(after, correct)

    def test_get_tuned_parameters(self):
        "test the get tuned parameters method"
        parameters = generate_regression_parameters_nd()
        regressor = LassoCVRegressor(parameters)

        params = regressor.get_tuned_parameters()

        correct = {'order': 1, 'terms': ['1', 'x1', 'x2', 'x3']}
        self.assertEqual(params, correct)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLassoCVRegressionSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)