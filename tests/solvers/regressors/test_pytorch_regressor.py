# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

r"test the multi layer perceptron regressor"

import tempfile
import unittest
from pathlib import Path
import datetime
import attrs
import shutil

import matplotlib
import numpy as np
import pandas as pd
import optuna
from optuna.distributions import IntDistribution, FloatDistribution

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.standard_scaler import StandardScaler
from gumps.solvers.regressors.regression_loader import load_regressor
from gumps.solvers.regressors.pytorch_regressor import (
    TorchMultiLayerPerceptronRegressionParameters,
    TorchMultiLayerPerceptronRegressor, OptunaParameters)
from gumps.solvers.regressors.regression_solver import RegressionParameters
from gumps.solvers.regressors.pytorch_utils.models import MLP
from gumps.solvers.regressors.regressor_data import DataRegression


def generate_regression_parameters(count: int = 5) -> TorchMultiLayerPerceptronRegressionParameters:
    "generate regression parameters"
    input_data = pd.DataFrame({"x": np.linspace(1, 2, count)})
    output_data = pd.DataFrame({"y": np.linspace(1, 2, count)})
    return TorchMultiLayerPerceptronRegressionParameters(input_data=input_data, output_data=output_data)


class TestTorchMultiLayerPerceptronRegressor(unittest.TestCase):
    "test the gaussian process regressor"

    def test_initialize_kernel(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        self.assertIsInstance(regressor.parameters, TorchMultiLayerPerceptronRegressionParameters)
        self.assertIsInstance(regressor.data_regression.input_scaler.scaler, StandardScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler.scaler, MinMaxScaler)
        self.assertIsInstance(regressor.data_regression.input_scaler, LogComboScaler)
        self.assertIsInstance(regressor.data_regression.output_scaler, LogComboScaler)
        self.assertIsInstance(regressor.regressor, MLP)
        self.assertIsInstance(regressor.data_regression, DataRegression)

    def test_torch_parameters_random(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        attrs.evolve(parameters, train_test_split="random")

    def test_torch_parameters_sequential(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        attrs.evolve(parameters, train_test_split="sequential")

    def test_torch_parameters_manual_exception(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.train_indices = None
            parameters.validation_indices = None
            attrs.evolve(parameters, train_test_split="manual")

    def test_torch_parameters_validation_fraction_less_than_zero(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = -0.1
            attrs.evolve(parameters, train_test_split="manual")

    def test_torch_parameters_validation_fraction_greater_than_one(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = 1.1
            attrs.evolve(parameters, train_test_split="manual")

    def test_torch_parameters_validation_fraction_zero(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = 0.0
            attrs.evolve(parameters, train_test_split="manual")

    def test_torch_parameters_validation_fraction_one(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.validation_fraction = 1.0
            attrs.evolve(parameters, train_test_split="manual")

    def test_torch_parameters_manual(self):
        "test the initialization of the kernel when it is None"
        parameters = generate_regression_parameters()
        attrs.evolve(parameters, train_test_split="manual",
                     train_indices=np.array([True, True, False, False]),
                     validation_indices=np.array([False, False, True, True]),
                     input_data=pd.DataFrame({"x": [1, 2, 3, 4]}),
                     output_data=pd.DataFrame({"y": [1, 2, 3, 4]}))

    def test_logging_directory(self):
        "test the logging directory"
        parameters = generate_regression_parameters()
        parameters.tensorboard_logging = True

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            parameters.logging_directory = temp_dir

            parameters.max_iter = 10
            regressor = TorchMultiLayerPerceptronRegressor(parameters)
            regressor.fit()

            self.assertTrue((temp_dir / "lightning_logs").exists())

    def test_split_sequential(self):
        "test setting the split to sequential"
        parameters = generate_regression_parameters()
        parameters.train_test_split = "sequential"

    def test_split_exception(self):
        "test setting the split to sequential"
        parameters = generate_regression_parameters()
        with self.assertRaises(ValueError):
            parameters.train_test_split = "foo"

    def test_get_regressor(self):
        "test the initialization of the regressor"
        parameters = generate_regression_parameters()
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        reg = regressor._get_regressor()
        self.assertIsInstance(reg, MLP)


    def test_get_regressor_with_parameters(self):
        "test the initialization of the regressor with parameters"
        parameters = generate_regression_parameters()
        parameters.hidden_layer_sizes = (10, 20)
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        reg = regressor._get_regressor()
        self.assertIsInstance(reg, MLP)
        self.assertEqual(reg.settings.hidden_layer_sizes, (10, 20))

    def test_get_scalers(self):
        "test the initialization of the input and output scalers"
        parameters = generate_regression_parameters()
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        input_scaler, output_scaler = regressor._get_scalers()
        self.assertIsInstance(input_scaler, LogComboScaler)
        self.assertIsInstance(output_scaler, LogComboScaler)

    def test_save(self):
        "test the save method"
        parameters = generate_regression_parameters(20)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            self.assertTrue((temp_dir / "model.pth").exists())
            self.assertTrue((temp_dir / "parameters.joblib").exists())
            self.assertTrue((temp_dir / "data").exists())

    def test_save_exception(self):
        "test the save method before fitting is done"
        parameters = generate_regression_parameters(20)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            with self.assertRaises(RuntimeError):
                regressor.save(temp_dir / "foo")

    def test_load(self):
        "test the load method"
        parameters = generate_regression_parameters(20)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)
            new_regressor = TorchMultiLayerPerceptronRegressor.load(temp_dir)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.scale_,
                            new_regressor.data_regression.input_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.var_,
                            new_regressor.data_regression.input_scaler.scaler.scaler.var_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.scale_,
                            new_regressor.data_regression.output_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.min_,
                            new_regressor.data_regression.output_scaler.scaler.scaler.min_)

            pd.testing.assert_frame_equal(regressor.predict(parameters.input_data),
                                        new_regressor.predict(parameters.input_data))

    def test_load_regressor(self):
        "test the load regressor method"
        parameters = generate_regression_parameters(20)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)

            instance = TorchMultiLayerPerceptronRegressor.__new__(TorchMultiLayerPerceptronRegressor)
            TorchMultiLayerPerceptronRegressor._load_regressor(instance, temp_dir)

            self.assertIsInstance(instance.regressor, MLP)
            self.assertIsInstance(instance.parameters, TorchMultiLayerPerceptronRegressionParameters)

    def test_load_instance_exception(self):
        "test the load instance method with an exception"
        base_dir = Path(__file__).parent.parent.parent / "data"
        regressor_dir = base_dir / "regressor" / "torch_example"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            #copy dir to temp_dir so that the original can't be modified
            shutil.copytree(regressor_dir, temp_dir, dirs_exist_ok=True)

            with self.assertRaises(RuntimeError):
                TorchMultiLayerPerceptronRegressor.load_instance(temp_dir)

    def test_generic_load(self):
        "test the load method"
        parameters = generate_regression_parameters(20)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            regressor.save(temp_dir)
            new_regressor = load_regressor(temp_dir)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.scale_,
                            new_regressor.data_regression.input_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.input_scaler.scaler.scaler.var_,
                            new_regressor.data_regression.input_scaler.scaler.scaler.var_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.scale_,
                            new_regressor.data_regression.output_scaler.scaler.scaler.scale_)

            self.assertEqual(regressor.data_regression.output_scaler.scaler.scaler.min_,
                            new_regressor.data_regression.output_scaler.scaler.scaler.min_)

            pd.testing.assert_frame_equal(regressor.predict(parameters.input_data),
                                        new_regressor.predict(parameters.input_data))


    def test_fit(self):
        "test the fit method"
        parameters = generate_regression_parameters(count=100)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

    def test_fit_sequential(self):
        "test the fit method with sequential split"
        parameters = generate_regression_parameters(count=100)
        parameters.train_test_split = "sequential"
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

    def test_error_metrics(self):
        "test the error metrics method"
        parameters = generate_regression_parameters(count=100)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()
        pd.testing.assert_series_equal(pd.Series({'score': 1.0, 'r2_score': 1.0,
                                                  'mean_squared_error': 0.0, 'mse': 0.0,
                                                  'root_mean_squared_error': 0.0, 'rmse': 0.0,
                                                  'normalized_mean_squared_error': 0.0,'nmse': 0.0,
                                                  'normalized_root_mean_squared_error': 0.0, 'nrmse': 0.0}),
                                       regressor.error_metrics(), atol=1e-1, rtol=1e-1)

    def test_predict(self):
        "test the predict method"
        parameters = generate_regression_parameters(count=1000)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        input_data = pd.DataFrame({'x': np.linspace(1, 2, 5)})
        output_data_correct = pd.DataFrame({'y': np.linspace(1, 2, 5)})

        output_data = regressor.predict(input_data)
        pd.testing.assert_frame_equal(output_data_correct, output_data, atol=1e-1, rtol=1e-1)

    def test_update_data(self):
        "test the update data method"
        parameters = generate_regression_parameters(count=50)
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        count = 5
        input_data = pd.DataFrame({"x": np.random.rand(count)})
        output_data = pd.DataFrame({"y": input_data.x})

        regressor.update_data(input_data, output_data)

        self.assertEqual(len(regressor.data_regression.split.full_input), 55)
        self.assertEqual(len(regressor.data_regression.split.full_output), 55)

        self.assertEqual(len(regressor.data_regression.scaled_split.full_input), 55)
        self.assertEqual(len(regressor.data_regression.scaled_split.full_output), 55)

    def test_plot_loss_curves(self):
        "test that the matplotlib is generated and that train and validation have the right number of entries"
        parameters = generate_regression_parameters(count=64)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        axes = regressor.plot_loss_curves()
        self.assertIsInstance(axes, matplotlib.axes._axes.Axes)

    def test_plot_loss_curves_exception(self):
        "test that the matplotlib is generated and that train and validation have the right number of entries"
        parameters = generate_regression_parameters(count=64)
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        with self.assertRaises(RuntimeError):
            regressor.plot_loss_curves()

    def test_save_loss_curves(self):
        "test saving the loss curve to a file"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            parameters = generate_regression_parameters(count=64)
            parameters.logging_directory = temp_dir
            parameters.max_iter = 10
            regressor = TorchMultiLayerPerceptronRegressor(parameters)
            regressor.fit()

            regressor.save_loss_curves(temp_dir / "loss_curves.png")

            self.assertTrue((temp_dir / "loss_curves.png").exists())

    def test_save_loss_curves_exception(self):
        "test that an exception is raised if called before fit"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            parameters = generate_regression_parameters(count=64)
            parameters.logging_directory = temp_dir
            parameters.max_iter = 10
            regressor = TorchMultiLayerPerceptronRegressor(parameters)

            with self.assertRaises(RuntimeError):
                regressor.save_loss_curves(temp_dir / "loss_curves.png")

    def test_auto_tune(self):
        "test the auto tune method runs without exception"
        parameters = generate_regression_parameters(count=100)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)

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
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            optuna_parameters = OptunaParameters(number_of_trials=10,
                                                 min_epochs=2,
                                                 storage=f"sqlite:///{temp_dir}/example.db")
            regressor.auto_tune(optuna_parameters)

            self.assertTrue((temp_dir / "example.db").exists())


    def test_get_tuned_parameters(self):
        "test the tune parameters method"
        parameters = generate_regression_parameters(count=100)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)

        output = regressor.get_tuned_parameters()

        correct ={'batch_size': 16,
                'learning_rate_init': 1e-3,
                'layer_sizes': [100,],
                'learning_rate_factor': 1e-1}

        self.assertEqual(output, correct)

    def test_clone_tune(self):
        "test the clone tune method can clone parameters and changes selected ones"
        parameters = generate_regression_parameters(count=100)
        parameters.max_iter = 10
        regressor = TorchMultiLayerPerceptronRegressor(parameters)

        trial = optuna.trial.FrozenTrial(number=4,
                    state=1,
                    values=[0.09923685980447434],
                    datetime_start=datetime.datetime(2023, 12, 8, 14, 15, 52, 609649),
                    datetime_complete=datetime.datetime(2023, 12, 8, 14, 15, 52, 844083),
                    params={'batch_size': 6,
                            'learning_rate_init': 0.003608899445512057,
                            'layers': 5,
                            'learning_rate_factor': 0.3823058915398655,
                            'layer_size_0': 6,
                            'layer_size_1': 4,
                            'layer_size_2': 1,
                            'layer_size_3': 1,
                            'layer_size_4': 3},
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
                    distributions={'batch_size': IntDistribution(high=9, log=False, low=6, step=1),
                                   'learning_rate_init': FloatDistribution(high=0.01, log=True, low=0.0001, step=None),
                                   'layers': IntDistribution(high=8, log=False, low=1, step=1),
                                   'learning_rate_factor': FloatDistribution(high=0.9, log=False, low=0.1, step=None),
                                   'layer_size_0': IntDistribution(high=9, log=False, low=1, step=1),
                                   'layer_size_1': IntDistribution(high=9, log=False, low=1, step=1),
                                   'layer_size_2': IntDistribution(high=9, log=False, low=1, step=1),
                                   'layer_size_3': IntDistribution(high=9, log=False, low=1, step=1),
                                   'layer_size_4': IntDistribution(high=9, log=False, low=1, step=1)},
                    trial_id=4, value=None)

        reg = regressor.clone_tune(trial)

        self.assertEqual(reg.parameters.batch_size, 2**6)
        self.assertEqual(reg.parameters.learning_rate_init, 0.003608899445512057)
        self.assertEqual(reg.parameters.learning_rate_factor, 0.3823058915398655)
        self.assertListEqual(reg.parameters.hidden_layer_sizes, [2**6, 2**4, 2**1, 2**1, 2**3])

    def test_get_split(self):
        "test the get split method"
        parameters = generate_regression_parameters(count=100)
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        split = regressor.get_split()
        self.assertTrue(len(split.train_input) == 90)
        self.assertTrue(len(split.validation_input) == 10)
        self.assertTrue(len(split.train_output) == 90)
        self.assertTrue(len(split.validation_output) == 10)

    def test_get_split_scaled(self):
        "test the get split method"
        parameters = generate_regression_parameters(count=100)
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        regressor.fit()

        split = regressor.get_split_scaled()
        self.assertTrue(len(split.train_input) == 90)
        self.assertTrue(len(split.validation_input) == 10)
        self.assertTrue(len(split.train_output) == 90)
        self.assertTrue(len(split.validation_output) == 10)

    def test_clone(self):
        "test the clone method"
        parameters = generate_regression_parameters(count=100)
        regressor = TorchMultiLayerPerceptronRegressor(parameters)
        clone = regressor.clone(parameters)
        self.assertEqual(clone.parameters, regressor.parameters)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTorchMultiLayerPerceptronRegressor)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestTorchMLP)
    unittest.TextTestRunner(verbosity=2).run(suite)