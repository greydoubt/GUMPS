# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"test the random forest regressor"

import tempfile
import unittest
from pathlib import Path
import attrs

import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import optuna

from gumps.solvers.regressors.xgboost_regressor import XGBoostRegressor, XGBoostParameters, OptunaParameters
from gumps.solvers.regressors.regression_loader import load_regressor
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.solvers.regressors.regressor_data import DataRegression

from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution


def generate_regression_parameters(count: int = 5):
    "generate regression parameters"
    input_data = pd.DataFrame({"x": np.linspace(0, 1, count)})
    output_data = pd.DataFrame({"y": np.linspace(0, 1, count)})
    return XGBoostParameters(input_data=input_data, output_data=output_data)

class TextXGBoostParameters(unittest.TestCase):
    "test the parameters"

    def test_initialize(self):
        "test the initialization"
        parameters = generate_regression_parameters()
        self.assertIsInstance(parameters, XGBoostParameters)

    def test_sequential(self):
        "test the sequential split"
        parameters = generate_regression_parameters()
        parameters.train_test_split = "sequential"


    def test_random(self):
        "test the random split"
        parameters = generate_regression_parameters()
        parameters.train_test_split = "random"

    def test_manual_exception(self):
        "test the manual split"
        parameters = generate_regression_parameters()

        with self.assertRaises(ValueError):
            parameters.train_indices = None
            parameters.validation_indices = None
            attrs.evolve(parameters, train_test_split="manual")

    def test_manual(self):
        "test the manual split"
        parameters = generate_regression_parameters()
        attrs.evolve(parameters, train_test_split="manual",
                     train_indices=[0, 1], validation_indices=[2, 3])

    def test_xgboost_parameters_validation_fraction_less_than_zero(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = -0.1
            attrs.evolve(parameters, train_test_split="manual")

    def test_xgboost_parameters_validation_fraction_greater_than_one(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = 1.1
            attrs.evolve(parameters, train_test_split="manual")

    def test_xgboost_parameters_validation_fraction_zero(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = 0.0
            attrs.evolve(parameters, train_test_split="manual")

    def test_xgboost_parameters_validation_fraction_one(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = 1.0
            attrs.evolve(parameters, train_test_split="manual")


    def test_invalid_split(self):
        "test the invalid split"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.train_test_split = "invalid"


class TestXGBoostRegressor(unittest.TestCase):
    "test the random forest regressor"

    def test_initialize(self):
        "test the initialization"
        parameters = generate_regression_parameters()
        regressor = XGBoostRegressor(parameters)
        self.assertIsInstance(regressor.parameters, XGBoostParameters)
        self.assertIsInstance(regressor.data_regression.input_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.input_scaler, LogComboScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler, LogComboScaler)
        self.assertIsInstance(regressor.regressor, xgb.XGBRegressor)
        self.assertIsInstance(regressor.data_regression, DataRegression)

    def test_get_regressor(self):
        "test the initialization of the regressor"
        parameters = generate_regression_parameters()
        regressor = XGBoostRegressor(parameters)
        reg = regressor._get_regressor()
        self.assertIsInstance(reg, xgb.XGBRegressor)

    def test_get_scalers(self):
        "test the initialization of the input and output scalers"
        parameters = generate_regression_parameters()
        regressor = XGBoostRegressor(parameters)
        input_scaler, output_scaler = regressor._get_scalers()
        self.assertIsInstance(input_scaler, LogComboScaler)
        self.assertIsInstance(output_scaler, LogComboScaler)

    def test_save(self):
        "test the save method"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            parameters = generate_regression_parameters()
            regressor = XGBoostRegressor(parameters)
            regressor.fit()
            regressor.save(temp_dir)

            self.assertTrue((temp_dir / "regressor.ubj").exists())
            self.assertTrue((temp_dir / "parameters.joblib").exists())
            self.assertTrue((temp_dir / "data").exists())


    def test_save_exception(self):
        "test that save fails when the simulation has not been fitted yet"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)

        with self.assertRaises(RuntimeError):
            regressor.save(Path("."))

    def test_load(self):
        "test the load method"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            parameters = generate_regression_parameters()
            regressor = XGBoostRegressor(parameters)
            regressor.fit()
            regressor.save(temp_dir)

            new_regressor = XGBoostRegressor.load(temp_dir)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.min_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.min_)

            pd.testing.assert_frame_equal(regressor.predict(regressor.parameters.input_data),
                                          new_regressor.predict(regressor.parameters.input_data))


    def test_generic_load(self):
        "test the load method"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            parameters = generate_regression_parameters()
            regressor = XGBoostRegressor(parameters)
            regressor.fit()
            regressor.save(temp_dir)

            new_regressor = XGBoostRegressor.load(temp_dir)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.input_scaler.scaler.scaler.min_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.scale_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.min_,
                             new_regressor.data_regression.output_scaler.scaler.scaler.min_)

            pd.testing.assert_frame_equal(regressor.predict(new_regressor.parameters.input_data),
                                          new_regressor.predict(new_regressor.parameters.input_data))

    def test_fit(self):
        "test the fit method"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)
        regressor.fit()

    def test_error_metrics(self):
        "test the error metrics method"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)
        regressor.fit()

        pd.testing.assert_series_equal(pd.Series({'score': 1.0, 'r2_score': 1.0,
                                            'mean_squared_error': 0.0, 'mse': 0.0,
                                            'root_mean_squared_error': 0.0, 'rmse': 0.0,
                                            'normalized_mean_squared_error': 0.0,'nmse': 0.0,
                                            'normalized_root_mean_squared_error': 0.0, 'nrmse': 0.0}),
                                regressor.error_metrics(), atol=1e-2, rtol=1e-2)

    def test_predict(self):
        "test the predict method"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)
        regressor.fit()

        input_data = pd.DataFrame({'x': np.linspace(0, 1, 5)})
        output_data_correct = pd.DataFrame({'y': np.linspace(0, 1, 5)})

        output_data = regressor.predict(input_data)
        pd.testing.assert_frame_equal(output_data_correct, output_data, atol=1e-1, rtol=1e-1)

    def test_update_data(self):
        "test the update data method"
        parameters = generate_regression_parameters(count=10)
        regressor = XGBoostRegressor(parameters)
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

        regressor = XGBoostRegressor(parameters)
        regressor.fit()

        regressor_2 = XGBoostRegressor(parameters)
        regressor_2.fit()

        pd.testing.assert_series_equal(regressor.error_metrics(),
                                       regressor_2.error_metrics())

    def test_auto_tune(self):
        "test the auto tune method runs without exception"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)

        regressor.fit()
        before = regressor.error_metrics()

        optuna_parameters = OptunaParameters(number_of_trials=10,
                                             min_epochs=2)
        regressor.auto_tune(optuna_parameters)

        after = regressor.error_metrics()

        #verify at least one number has changed
        self.assertTrue( (before != after).any())

    def test_auto_tune_storage(self):
        "test the auto tune method runs without exception"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            optuna_parameters = OptunaParameters(number_of_trials=10,
                                                 min_epochs=2,
                                                 storage=f"sqlite:///{temp_dir}/example.db")
            regressor.auto_tune(optuna_parameters)

            self.assertTrue((temp_dir / "example.db").exists())

    def test_get_tuned_parameters(self):
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)

        output = regressor.get_tuned_parameters()

        correct = {'booster': 'gbtree',
                     'gamma': None,
                     'max_depth': None,
                     'min_child_weight': None,
                     'n_estimators': 100,
                     'num_parallel_tree': None,
                     'learning_rate': None}

        self.assertEqual(output, correct)


    def test_clone_tune(self):
        "test the clone tune method can clone parameters and changes selected ones"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)

        trial = optuna.trial.FrozenTrial(number=4,
                    state=1,
                    values=[0.09923685980447434],
                    datetime_start=datetime.datetime(2023, 12, 8, 14, 15, 52, 609649),
                    datetime_complete=datetime.datetime(2023, 12, 8, 14, 15, 52, 844083),
                    params={'booster': 'gbtree',
                            'gamma': 1e-1,
                            'max_depth': 3,
                            'min_child_weight': 1e-1,
                            'n_estimators': 10,
                            'num_parallel_tree': 1,
                            'learning_rate': 1e-1},
                    user_attrs={},
                    system_attrs={'completed_rung_0': -1.3527949787173643,
                                  'completed_rung_1': -2.539117119246736},
                    intermediate_values={0: -1.143145199345226,
                                         1: -1.2429877481763745,
                                         2: -1.3527949787173643,
                                         3: -1.4781147633603064,
                                         4: -1.621041171751107,
                                         5: -1.785964762939748,
                                         6: -1.9821591802503538,
                                         7: -2.2243583077491103,
                                         8: -2.539117119246736,
                                         9: -2.9737313289149934},
                    distributions={'booster': CategoricalDistribution(choices=('gbtree', 'gblinear', 'dart')),
                                   'gamma': FloatDistribution(high=1e-2, log=True, low=1e-8, step=None),
                                   'max_depth': IntDistribution(high=10, log=False, low=1, step=1),
                                   'min_child_weight': FloatDistribution(high=1e2, log=True, low=1e-6, step=None),
                                   'n_estimators': IntDistribution(high=1000, log=False, low=10, step=1),
                                   'num_parallel_tree': IntDistribution(high=10, log=False, low=1, step=1),
                                   'learning_rate': FloatDistribution(high=1.0, log=True, low=0.001, step=None)},
                    trial_id=4, value=None)

        reg = regressor.clone_tune(trial)

        self.assertEqual(reg.parameters.booster, 'gbtree')
        self.assertEqual(reg.parameters.gamma, 1e-1)
        self.assertEqual(reg.parameters.max_depth, 3)
        self.assertEqual(reg.parameters.min_child_weight, 1e-1)
        self.assertEqual(reg.parameters.n_estimators, 10)
        self.assertEqual(reg.parameters.num_parallel_tree, 1)
        self.assertEqual(reg.parameters.learning_rate, 1e-1)

    def test_clone(self):
        "test the clone method"
        parameters = generate_regression_parameters(count=100)
        regressor = XGBoostRegressor(parameters)
        reg = regressor.clone(parameters)
        self.assertEqual(reg.parameters, regressor.parameters)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestXGBoostRegressor)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TextXGBoostParameters)
    unittest.TextTestRunner(verbosity=2).run(suite)
