# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a multi-layer perceptron regressor with input and output scaling."""

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from pathlib import Path
import tempfile

import attrs
import joblib
import matplotlib.axes
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from gumps.solvers.regressors.pytorch_utils.loaders import DataModule, DataModuleParameters, DataModulePredict
from gumps.solvers.regressors.pytorch_utils.models import MLP, MLPParameters
from gumps.solvers.regressors.regressor_data import Split

#pytorch defaults to float32, but we want float64
torch.set_default_dtype(torch.float64)

#tensor core support for models
torch.set_float32_matmul_precision('medium')

from typing import Callable
import optuna

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.standard_scaler import StandardScaler
import gumps.solvers.regressors.regression_solver as regression_solver
from gumps.solvers.regressors.error_metrics import ErrorMetrics
from gumps.common.logging import LoggingContext

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

OptunaParameters = regression_solver.OptunaParameters


@attrs.define(kw_only=True)
class TorchMultiLayerPerceptronRegressionParameters(regression_solver.RegressionParameters):
    "Parameters for the multi-layer perceptron regressor."
    hidden_layer_sizes: list[int] = [100,]
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.SiLU
    learning_rate: str = "constant"
    learning_rate_init: float = 0.001
    learning_rate_patience:int = 10
    learning_rate_factor:float = 0.1
    batch_size: int = 16
    random_state: int | None = None
    shuffle: bool = True
    max_iter: int = 400
    early_stopping_patience: int = 10
    tensorboard_logging: bool = False
    enable_progress_bar: bool = True
    logging_directory: Path | None = None
    validation_fraction: float = attrs.field(default=0.1)
    @validation_fraction.validator
    def _check_validation_fraction(self, _, value):
        if not (0 < value < 1):
            raise ValueError(f"validation_fraction must be between 0 and 1, not {value}")

    def get_mlp_parameters(self) -> MLPParameters:
        "construct and return MLPParameters"
        return MLPParameters(
            input_dim=self.input_data.shape[1],
            output_dim=self.output_data.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            learning_rate_patience=self.learning_rate_patience,
            learning_rate_factor=self.learning_rate_factor,
        )

    def get_data_module_parameters(self) -> DataModuleParameters:
        "construct and return DataModuleParameters"
        return DataModuleParameters(
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

class TorchMultiLayerPerceptronRegressor(regression_solver.AbstractRegressor):
    "Implement a multi-layer perceptron regressor with input and output scaling."

    def __init__(self, parameters: TorchMultiLayerPerceptronRegressionParameters, skip_init:bool=False):
        "Initialize the solver."
        self.parameters: TorchMultiLayerPerceptronRegressionParameters
        self.error_metrics_data: ErrorMetrics = ErrorMetrics()

        if not skip_init:
            super().__init__(parameters)

    def clone(self, parameters: TorchMultiLayerPerceptronRegressionParameters, skip_init:bool=False) -> 'TorchMultiLayerPerceptronRegressor':
        "Clone the solver."
        return TorchMultiLayerPerceptronRegressor(parameters, skip_init=skip_init)

    def _get_scalers(self) -> tuple[LogComboScaler, LogComboScaler]:
        """Return the input and output scalers, For regression neural networks
        use a standard scaler for the input and a min max scaler for the output."""
        return LogComboScaler(StandardScaler()), LogComboScaler(MinMaxScaler())


    def _get_regressor(self) -> MLP:
        "Return the regressor."
        return MLP(self.parameters.get_mlp_parameters())


    def _get_datamodule(self) -> DataModule:
        "return the data module"
        return DataModule(scaled_split = self.data_regression.scaled_split,
                          parameters = self.parameters.get_data_module_parameters())

    def _fit(self, additional_callbacks:list[Callable]|None=None):
        "fit the regressor"
        #train_dataloader, val_dataloader = self._get_dataloaders()
        data = self._get_datamodule()

        early_stop_callback = EarlyStopping(
            min_delta=1e-3,
            monitor="val_log_loss",
            patience=self.parameters.early_stopping_patience,
            mode='min'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_callback = ModelCheckpoint(
                dirpath=tmpdir,
                save_top_k=1,
                monitor="val_log_loss",
                mode="min")


            callbacks = [early_stop_callback, checkpoint_callback]

            if additional_callbacks is not None:
                callbacks.extend(additional_callbacks)

            trainer = pl.Trainer(max_epochs=self.parameters.max_iter,
                                callbacks=callbacks,
                                enable_checkpointing=True,
                                logger=self.parameters.tensorboard_logging,
                                enable_progress_bar=self.parameters.enable_progress_bar,
                                default_root_dir=self.parameters.logging_directory)

            trainer.fit(self.regressor, datamodule=data)

            reg = MLP.load_from_checkpoint(checkpoint_callback.best_model_path, settings=self.regressor.settings)

        reg.train_losses = self.regressor.train_losses
        reg.validation_losses = self.regressor.validation_losses
        self.regressor = reg
        self.fitted = True
        self.update_error_metrics()

    def clone_tune(self, trial:optuna.trial.Trial) -> 'TorchMultiLayerPerceptronRegressor':
        "Create a new regressor with the same parameters and scalers as the original."
        batch_size = trial.suggest_int('batch_size', 6, 9)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        layers = trial.suggest_int('layers', 1, 8)
        learning_rate_factor = trial.suggest_float('learning_rate_factor', 0.1, 0.9)

        layer_size = []
        for layer in range(layers):
            layer_size.append(2**trial.suggest_int(f'layer_size_{layer}', 1, 9))

        reg_data = attrs.evolve(self.parameters,
                                learning_rate_factor=learning_rate_factor,
                                learning_rate_init=learning_rate_init,
                                batch_size=2**batch_size,
                                hidden_layer_sizes=layer_size)

        reg = self.clone(reg_data, skip_init=True)
        reg.parameters = reg_data
        reg.data_regression = self.data_regression
        reg.regressor = reg._get_regressor()
        reg.fitted = False
        return reg

    def get_tuned_parameters(self) -> dict:
        "return the current values of the parameters that can be tuned"
        temp =  {'batch_size': self.parameters.batch_size,
                'learning_rate_init': self.parameters.learning_rate_init,
                'layer_sizes': self.parameters.hidden_layer_sizes,
                'learning_rate_factor': self.parameters.learning_rate_factor}
        return temp

    def auto_tune(self, settings:OptunaParameters) -> None:
        "this will auto tune the regressor, it may have problems inside a jupyter notebook"
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=settings.min_epochs,
                                                        reduction_factor=4,
                                                        min_early_stopping_rate=0,
                                                        bootstrap_count=0)

        study = optuna.create_study(direction='minimize',
                            storage=settings.storage,
                            pruner=pruner)

        def objective(trial:optuna.trial.Trial):
            callbacks = [optuna.integration.PyTorchLightningPruningCallback(trial,
                                                                            monitor="val_log_loss")]
            reg = self.clone_tune(trial)
            reg._fit(additional_callbacks=callbacks)
            return reg.error_metrics()['nrmse']

        study.optimize(objective,
                       n_trials=settings.number_of_trials)

        best_reg = self.clone_tune(study.best_trial)
        self.parameters = best_reg.parameters
        self.regressor = best_reg.regressor
        self.fit()


    def _predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "predict the output data"
        input_data_scaled = self.data_regression.input_scaler.transform(input_data)

        data = DataModulePredict(input_data = input_data_scaled,
                          parameters = self.parameters.get_data_module_parameters())

        with LoggingContext(logging.getLogger("pytorch_lightning"), level=logging.ERROR):
            trainer = pl.Trainer(logger=False,
                                enable_progress_bar=False,
                                enable_model_summary=False,
                                inference_mode=True)

            torch_data: list[torch.Tensor] = trainer.predict(self.regressor, datamodule=data)

        output_data_numpy = np.vstack([i.cpu().detach().numpy() for i in torch_data])

        output_data_scaled = pd.DataFrame(output_data_numpy, columns=self.parameters.output_data.columns)

        output_data = self.data_regression.output_scaler.inverse_transform(output_data_scaled)

        return output_data

    def save(self, path_dir: Path) -> None:
        "save the regressor, input scaler and output scaler"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted yet")

        self.error_metrics_data.save(path_dir)
        torch.save(self.regressor.state_dict(), path_dir / 'model.pth')
        joblib.dump(self.parameters, path_dir / "parameters.joblib")
        with open(path_dir / "regressor.txt", "w", encoding='utf-8') as f:
            f.write(self.__class__.__name__)

        self.data_regression.save(path_dir)


    @classmethod
    def _load_regressor(cls, instance, path_dir:Path):
        "load the regressor and raise an exception if it fails to load"
        instance.parameters = joblib.load(path_dir / "parameters.joblib")

        #have to initialize the regressor before loading the state dict
        instance.regressor = instance._get_regressor()

        instance.regressor.load_state_dict(torch.load(path_dir / 'model.pth'))

        #this is needed so that older objects can be loaded without breaking
        if (path_dir / "train_indices.joblib").exists():
            instance.train_indices = joblib.load(path_dir / "train_indices.joblib")

        if (path_dir / "validation_indices.joblib").exists():
            instance.validation_indices = joblib.load(path_dir / "validation_indices.joblib")


    def plot_loss_curves(self) -> matplotlib.axes._axes.Axes:
        "plot loss curve"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted yet")

        return self.regressor.plot_loss_curves()

    def save_loss_curves(self, path:Path):
        "save the loss curves"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted yet")

        axes = self.regressor.plot_loss_curves()
        fig = axes.get_figure()
        fig.savefig(path)
        plt.close(fig)


    def get_split(self) -> Split:
        "return the split"
        return self.data_regression.split


    def get_split_scaled(self) -> Split:
        "return the split"
        return self.data_regression.scaled_split
