# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the random forest regressor"

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.ensemble

from gumps.solvers.regressors.random_forest_regressor import \
    RandomForestRegressor, RandomForestRegressionParameters
from gumps.solvers.regressors.regression_loader import load_regressor
from gumps.solvers.regressors.regression_solver import RegressionParameters
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.solvers.regressors.regressor_data import DataRegression


def generate_regression_parameters(count: int = 5):
    "generate regression parameters"
    input_data = pd.DataFrame({"x": np.linspace(1, 10, count)})
    output_data = pd.DataFrame({"y": np.linspace(1, 10, count)})
    return RandomForestRegressionParameters(input_data=input_data, output_data=output_data)

class TestRandomForestRegressionSolver(unittest.TestCase):
    "test the random forest regressor"

    def test_initialize(self):
        "test the initialization"
        parameters = generate_regression_parameters()
        regressor = RandomForestRegressor(parameters)
        self.assertIsInstance(regressor.parameters, RandomForestRegressionParameters)
        self.assertIsInstance(regressor.data_regression.input_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.input_scaler, LogComboScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler, LogComboScaler)
        self.assertIsInstance(regressor.regressor, sklearn.ensemble.RandomForestRegressor)
        self.assertIsInstance(regressor.data_regression, DataRegression)

    def test_get_regressor(self):
        "test the initialization of the regressor"
        parameters = generate_regression_parameters()
        regressor = RandomForestRegressor(parameters)
        reg = regressor._get_regressor()
        self.assertIsInstance(reg, sklearn.ensemble.RandomForestRegressor)

    def test_get_scalers(self):
        "test the initialization of the input and output scalers"
        parameters = generate_regression_parameters()
        regressor = RandomForestRegressor(parameters)
        input_scaler, output_scaler = regressor._get_scalers()
        self.assertIsInstance(input_scaler, LogComboScaler)
        self.assertIsInstance(output_scaler, LogComboScaler)

    def test_save(self):
        "test the save method"
        parameters = generate_regression_parameters()
        regressor = RandomForestRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            self.assertTrue((temp_dir / "regressor.joblib").exists())
            self.assertTrue((temp_dir / "parameters.joblib").exists())
            self.assertTrue((temp_dir / "data").exists())

    def test_load(self):
        "test the load method"
        parameters = generate_regression_parameters()
        regressor = RandomForestRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)
            new_regressor = RandomForestRegressor.load(temp_dir)

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
        parameters = generate_regression_parameters()
        regressor = RandomForestRegressor(parameters)
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
        regressor = RandomForestRegressor(parameters)
        regressor.fit()

    def test_error_metrics(self):
        "test the error metrics method"
        parameters = generate_regression_parameters(count=1000)
        regressor = RandomForestRegressor(parameters)
        regressor.fit()
        pd.testing.assert_series_equal(pd.Series({'score': 1.0, 'r2_score': 1.0,
                                            'mean_squared_error': 0.0, 'mse': 0.0,
                                            'root_mean_squared_error': 0.0, 'rmse': 0.0,
                                            'normalized_mean_squared_error': 0.0,'nmse': 0.0,
                                            'normalized_root_mean_squared_error': 0.0, 'nrmse': 0.0}),
                                regressor.error_metrics(), atol=1e-2, rtol=1e-2)

    def test_predict(self):
        "test the predict method"
        parameters = generate_regression_parameters(count=1000)
        regressor = RandomForestRegressor(parameters)
        regressor.fit()

        input_data = pd.DataFrame({'x': np.linspace(1, 10, 5)})
        output_data_correct = pd.DataFrame({'y': np.linspace(1, 10, 5)})

        output_data = regressor.predict(input_data)
        pd.testing.assert_frame_equal(output_data_correct, output_data, atol=1e-1, rtol=1e-1)

    def test_update_data(self):
        "test the update data method"
        parameters = generate_regression_parameters(count=10)
        regressor = RandomForestRegressor(parameters)
        regressor.fit()

        count = 5
        input_data = pd.DataFrame({"x": np.random.rand(count)})
        output_data = pd.DataFrame({"y": input_data.x})

        regressor.update_data(input_data, output_data)

        self.assertEqual(len(regressor.data_regression.split.full_input), 15)
        self.assertEqual(len(regressor.data_regression.split.full_output), 15)

        self.assertEqual(len(regressor.data_regression.scaled_split.full_input), 15)
        self.assertEqual(len(regressor.data_regression.scaled_split.full_output), 15)


    def test_random_state(self):
        "test the random state and verify that it is deterministic"
        parameters = generate_regression_parameters(count=100)
        parameters.random_state = 10

        regressor = RandomForestRegressor(parameters)
        regressor.fit()

        regressor_2 = RandomForestRegressor(parameters)
        regressor_2.fit()

        pd.testing.assert_series_equal(regressor.error_metrics(),
                                       regressor_2.error_metrics())


    def test_load_rebuild(self):
        "load the regressor with a different version of sklearn than it was saved with and auto-matically rebuild it"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "random_forest"

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
        regressor_dir = base_dir / "regressor" / "random_forest"

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
        regressor_dir = base_dir / "regressor" / "random_forest"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                load_regressor(temp_dir, auto_rebuild=False)

    def test_clone(self):
        "test the clone method"
        parameters = generate_regression_parameters(count=100)
        regressor = RandomForestRegressor(parameters)
        new_regressor = regressor.clone(parameters)
        self.assertEqual(regressor.parameters, new_regressor.parameters)

    def test_clone_tune(self):
        "test the clone tune method"
        parameters = generate_regression_parameters(count=100)
        regressor = RandomForestRegressor(parameters)
        with self.assertRaises(NotImplementedError):
            regressor.clone_tune(None)

    def test_auto_tune(self):
        "test the auto tune method"
        parameters = generate_regression_parameters(count=100)
        regressor = RandomForestRegressor(parameters)
        with self.assertRaises(NotImplementedError):
            regressor.auto_tune(None)

    def test_get_tuned_parameters(self):
        "test the get tuned parameters method"
        parameters = generate_regression_parameters(count=100)
        regressor = RandomForestRegressor(parameters)
        with self.assertRaises(NotImplementedError):
            regressor.get_tuned_parameters()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomForestRegressionSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)
