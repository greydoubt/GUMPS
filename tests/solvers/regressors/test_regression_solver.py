# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test update base regression solver"

import shutil
import tempfile
import unittest
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.exceptions

from gumps.solvers.regressors.regression_solver import RegressionParameters
from gumps.solvers.regressors.trivial_regressor import TrivialLinearRegressor
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.log_combo_scaler import LogComboScaler

from gumps.solvers.regressors.error_metrics import ErrorMetrics
from gumps.solvers.regressors.regressor_data import DataRegression


def generate_regression_parameters(count: int = 5):
    "generate regression parameters"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
    return RegressionParameters(input_data=input_data, output_data=output_data)


def processing_function(output_data: pd.DataFrame) -> pd.DataFrame:
    "process the input data"
    return output_data

class TestRegressionSolver(unittest.TestCase):
    "test the regressor"

    def test_initialize(self):
        "test the initialization"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        self.assertIsInstance(regressor.parameters, RegressionParameters)
        self.assertIsInstance(regressor.data_regression.input_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.input_scaler, LogComboScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler, LogComboScaler)
        self.assertIsInstance(regressor.regressor, sklearn.linear_model.LinearRegression)
        self.assertIsInstance(regressor.data_regression, DataRegression)


    def test_get_scalers(self):
        "test the initialization of the input and output scalers"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        input_scaler, output_scaler = regressor._get_scalers()
        self.assertIsInstance(input_scaler, LogComboScaler)
        self.assertIsInstance(output_scaler, LogComboScaler)

    def test_get_regressor(self):
        "test the initialization of the regressor"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        reg = regressor._get_regressor()
        self.assertIsInstance(reg, sklearn.linear_model.LinearRegression)

    def test_save_exception(self):
        "test the save method with an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            with self.assertRaises(RuntimeError):
                regressor.save(temp_dir)

    def test_save(self):
        "test the save method"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            self.assertTrue((temp_dir / "regressor.joblib").exists())
            self.assertTrue((temp_dir / "parameters.joblib").exists())
            self.assertTrue((temp_dir / "data").exists())


    def test_load_error_metrics_data(self):
        "test loading the error metrics data"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            metrics = regressor.error_metrics_data.get_metrics_frame()

            regressor.load_error_metrics_data(temp_dir)

            metrics_load = regressor.error_metrics_data.get_metrics_frame()

            self.assertIsInstance(regressor.error_metrics_data, ErrorMetrics)
            pd.testing.assert_frame_equal(metrics, metrics_load)

    def test_load_error_metrics_data_exception(self):
        "test loading the error metrics data with an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            metrics = regressor.error_metrics_data.get_metrics_frame()

            (temp_dir / "error_scalar.joblib").unlink()
            (temp_dir / "error_vector.joblib").unlink()

            regressor.load_error_metrics_data(temp_dir)

            metrics_load = regressor.error_metrics_data.get_metrics_frame()

            self.assertIsInstance(regressor.error_metrics_data, ErrorMetrics)
            pd.testing.assert_frame_equal(metrics, metrics_load)

    def test_contextmanager(self):
        "test the context manager interface"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with regressor:
            pass


    def test_contextmanager_not_fitted(self):
        "test the context manager interface"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)

        with self.assertRaises(RuntimeError):
            with regressor:
                pass


    def test_start(self):
        "just make sure start does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()
        regressor.start()


    def test_start_not_fitted(self):
        "just make sure start does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)

        with self.assertRaises(RuntimeError):
            regressor.start()


    def test_stop(self):
        "just make sure stop does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.stop()


    def test_run(self):
        "just make sure run does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        output_data = regressor.run(input_data = parameters.input_data, processing_function = processing_function)

        pd.testing.assert_frame_equal(output_data, parameters.output_data, atol=1e-2, rtol=1e-2)

    def test_iter_run(self):
        "just make sure iter_run does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        output_data = regressor.iter_run(input_data = parameters.input_data, processing_function = processing_function)

        for idx, _, data in output_data:
            out = parameters.output_data.iloc[idx]
            out.name = None
            pd.testing.assert_series_equal(data, out, atol=1e-2, rtol=1e-2)


    def test_run_not_fitted(self):
        "just make sure run does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)

        with self.assertRaises(RuntimeError):
            regressor.run(input_data = parameters.input_data, processing_function = processing_function)


    def test_full_results(self):
        "just make sure full_results does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        regressor.run(parameters.input_data, processing_function)

        input_data, results = regressor.full_results()

        pd.testing.assert_frame_equal(input_data, parameters.input_data, atol=1e-2, rtol=1e-2)
        pd.testing.assert_frame_equal(results, parameters.output_data, atol=1e-2, rtol=1e-2)


    def test_save_results(self):
        "just make sure save_results does not raise an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)

        regressor.save_results(parameters.input_data, parameters.output_data)

        pd.testing.assert_frame_equal(regressor.input_data, parameters.input_data, atol=1e-2, rtol=1e-2)
        pd.testing.assert_frame_equal(regressor.results, parameters.output_data, atol=1e-2, rtol=1e-2)


    def test_load(self):
        "test the load method"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)
            new_regressor = TrivialLinearRegressor.load(temp_dir)

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

    def test_load_missing_error_metrics(self):
        "test loading the error metrics data with an exception"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            (temp_dir / "error_scalar.joblib").unlink()
            (temp_dir / "error_vector.joblib").unlink()

            instance = TrivialLinearRegressor.load(temp_dir)

            self.assertIsInstance(instance.error_metrics_data, ErrorMetrics)

    def test_fit(self):
        "test the fit method"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

    def test_error_metrics_exceptions(self):
        "test the error metrics method exceptions"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        with self.assertRaises(RuntimeError):
            regressor.error_metrics()

    def test_error_metrics(self):
        "test the error metrics method"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()
        pd.testing.assert_series_equal(pd.Series({'score': 1.0, 'r2_score': 1.0,
                                                  'mean_squared_error': 0.0, 'mse': 0.0,
                                                  'root_mean_squared_error': 0.0, 'rmse': 0.0,
                                                  'normalized_mean_squared_error': 0.0,'nmse': 0.0,
                                                  'normalized_root_mean_squared_error': 0.0, 'nrmse': 0.0}),
                                       regressor.error_metrics(), atol=1e-3, rtol=1e-3)

    def test_error_metrics_multioutput(self):
        "test the error metrics multioutput method"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()
        pd.testing.assert_series_equal(pd.Series({'score': 1.0, 'r2_score': 1.0,
                                                  'mean_squared_error':[0.0], 'mse': [0.0],
                                                  'root_mean_squared_error': [0.0], 'rmse': [0.0],
                                                  'normalized_mean_squared_error': [0.0],'nmse': [0.0],
                                                  'normalized_root_mean_squared_error': [0.0], 'nrmse': [0.0]}),
                                       regressor.error_metrics(multioutput="raw_values"), atol=1e-3, rtol=1e-3)

    def test_error_metrics_multioutput_error(self):
        "test the error metrics multioutput method error"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()
        with self.assertRaises(ValueError):
            regressor.error_metrics(multioutput="raw_v")

    def test_error_frame(self):
        "test the error frame method"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        correct = pd.DataFrame({'score':{'y': 1.0},
                                'r2_score':{'y': 1.0},
                                'mean_squared_error':{'y': 0.0},
                                'mse':{'y': 0.0},
                                'root_mean_squared_error':{'y': 0.0},
                                'rmse':{'y': 0.0},
                                'normalized_mean_squared_error':{'y': 0.0},
                                'nmse':{'y': 0.0},
                                'normalized_root_mean_squared_error':{'y': 0.0},
                                'nrmse':{'y': 0.0}}).T

        pd.testing.assert_frame_equal(correct, regressor.error_frame(), atol=1e-3, rtol=1e-3)

    def test_error_frame_exception(self):
        "test the error frame method exception"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        with self.assertRaises(RuntimeError):
            regressor.error_frame()

    def test_predict_exception(self):
        "test the predict method exceptions"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        with self.assertRaises(RuntimeError):
            regressor.predict(pd.DataFrame({'x': [1.0]}))

    def test_predict(self):
        "test the predict method"
        parameters = generate_regression_parameters(count=100)
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        input_data = pd.DataFrame({'x': np.linspace(0, 1, 5)})
        output_data_correct = pd.DataFrame({'y': np.linspace(0, 1, 5)})

        output_data = regressor.predict(input_data)
        pd.testing.assert_frame_equal(output_data_correct, output_data, atol=1e-2, rtol=1e-2)

    def test_update_data(self):
        "test the update data method"
        parameters = generate_regression_parameters(count=10)
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        count = 5
        input_data = pd.DataFrame({"x": np.random.rand(count)})
        output_data = pd.DataFrame({"y": input_data.x})

        regressor.update_data(input_data, output_data)

        self.assertEqual(len(regressor.data_regression.split.full_input), 15)
        self.assertEqual(len(regressor.data_regression.split.full_output), 15)

        self.assertEqual(len(regressor.data_regression.scaled_split.full_input), 15)
        self.assertEqual(len(regressor.data_regression.scaled_split.full_output), 15)


    def test_load_regressor(self):
        "test loading the regressor"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            instance = TrivialLinearRegressor.__new__(TrivialLinearRegressor)

            instance._load_regressor(instance, temp_dir)

            self.assertIsInstance(instance.regressor, sklearn.linear_model.LinearRegression)


    def test_load_regressor_exceptions(self):
        "test loading the regressor with an exception"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "trivial"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                TrivialLinearRegressor.load_instance(temp_dir)


    def test_load_instance(self):
        "test loading the regressor"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            instance = TrivialLinearRegressor.load_instance(temp_dir)

            self.assertIsInstance(instance.regressor, sklearn.linear_model.LinearRegression)

    def test_update_load_indices(self):
        "test the update load indices method"
        parameters = generate_regression_parameters(count=10)
        regressor = TrivialLinearRegressor(parameters)

        train_indices = list(np.random.choice(10, 5, replace=False))
        validation_indices = list(np.random.choice(10, 5, replace=False))

        regressor.train_indices = train_indices
        regressor.validation_indices = validation_indices

        regressor.update_load_indices()

        self.assertListEqual(list(regressor.parameters.train_indices), train_indices)
        self.assertListEqual(list(regressor.parameters.validation_indices), validation_indices)


    def test_load_instance_missing_data_regression(self):
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            shutil.copy(temp_dir / "data" / "input_scaler.joblib", temp_dir / "input_scaler.joblib")
            shutil.copy(temp_dir / "data" / "output_scaler.joblib", temp_dir / "output_scaler.joblib")
            shutil.rmtree(temp_dir / "data")

            instance = TrivialLinearRegressor.load_instance(temp_dir)

            self.assertIsInstance(instance.regressor, sklearn.linear_model.LinearRegression)

    def test_load_missing_data_regression(self):
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            shutil.copy(temp_dir / "data" / "input_scaler.joblib", temp_dir / "input_scaler.joblib")
            shutil.copy(temp_dir / "data" / "output_scaler.joblib", temp_dir / "output_scaler.joblib")
            shutil.rmtree(temp_dir / "data")

            instance = TrivialLinearRegressor.load(temp_dir)

            self.assertIsInstance(instance.regressor, sklearn.linear_model.LinearRegression)


    def test_load_instance_exception(self):
        "test the load instance method with an exception"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "trivial"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                TrivialLinearRegressor.load_instance(temp_dir)


    def test_load_rebuild(self):
        "load the regressor with a different version of sklearn than it was saved with and auto-matically rebuild it"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "trivial"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.regressors.regression_solver', level='WARNING') as cm:
                TrivialLinearRegressor.load(temp_dir, auto_resave=False)
            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor failed to load, rebuilding regressor.' in cm.output)

    def test_load_rebuild_resave(self):
        "load the regressor with a different version of sklearn than it was saved with and auto-matically rebuild it"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "trivial"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.regressors.regression_solver', level='WARNING') as cm:
                TrivialLinearRegressor.load(temp_dir, auto_resave=True, auto_rebuild=True)
            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor failed to load, rebuilding regressor.' in cm.output)
            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor was rebuilt, resaving regressor.' in cm.output)

    def test_load_exception(self):
        "load the regressor with a different version of sklearn and disallow auto-rebuild"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "trivial"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                TrivialLinearRegressor.load(temp_dir, auto_rebuild=False)


    def test_rebuild_model(self):
        "rebuild the regressor"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "trivial"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.regressors.regression_solver', level='WARNING') as cm:
                instance = TrivialLinearRegressor.rebuild_model(temp_dir, auto_resave=False)

            self.assertIsInstance(instance.regressor, sklearn.linear_model.LinearRegression)
            self.assertIsInstance(instance, TrivialLinearRegressor)

            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor failed to load, rebuilding regressor.' in cm.output)

    def test_rebuild_model_resave(self):
        "rebuild the regressor and resave it"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "trivial"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertLogs('gumps.solvers.regressors.regression_solver', level='WARNING') as cm:
                instance = TrivialLinearRegressor.rebuild_model(temp_dir, auto_resave=True)

            self.assertIsInstance(instance.regressor, sklearn.linear_model.LinearRegression)
            self.assertIsInstance(instance, TrivialLinearRegressor)

            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor failed to load, rebuilding regressor.' in cm.output)
            self.assertTrue('WARNING:gumps.solvers.regressors.regression_solver:Regressor was rebuilt, resaving regressor.' in cm.output)

    def test_clone(self):
        "test the clone method"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        clone = regressor.clone(parameters)
        self.assertEqual(regressor.parameters, clone.parameters)

    def test_auto_tune(self):
        "test the auto tune method"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        with self.assertRaises(NotImplementedError):
            regressor.auto_tune(None)

    def test_get_tuned_parameters(self):
        "test the get tuned parameters method"
        parameters = generate_regression_parameters()
        regressor = TrivialLinearRegressor(parameters)
        with self.assertRaises(NotImplementedError):
            regressor.get_tuned_parameters()


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRegressionSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)
