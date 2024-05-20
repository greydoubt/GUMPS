# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a xgboost regressor with input and output scaling"""

from pathlib import Path

import attrs
import joblib
import pandas as pd
import xgboost as xgb

import gumps.solvers.regressors.regression_solver as regression_solver
from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.minmax_scaler import MinMaxScaler
from typing import Callable
import optuna
from gumps.solvers.regressors.error_metrics import ErrorMetrics

OptunaParameters = regression_solver.OptunaParameters

@attrs.define(kw_only=True)
class XGBoostParameters(regression_solver.RegressionParameters):
    "xgboost regression parameters"
    objective: str = "reg:squarederror"
    n_estimators: int = 100
    max_depth: int | None = None
    max_leaves: int | None = None
    max_bin: int | None = None
    grow_policy: str | None = None
    learning_rate: float | None = None
    booster: str = "gbtree"
    verbosity: int | None = None
    fit_verbosity: bool | int | None = None
    tree_method: str | None = None
    n_jobs: int | None = None
    gamma: float | None = None
    min_child_weight: float | None = None
    max_delta_step: float | None = None
    subsample: float | None = None
    colsample_bytree: float | None = None
    colsample_bylevel: float | None = None
    colsample_bynode: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None
    scale_pos_weight: float | None = None
    base_score: float | None = None
    random_state: int | None = None
    num_parallel_tree: int | None = None
    early_stopping_rounds: int = 10
    eval_metric: str = "rmse"
    validation_fraction: float = attrs.field(default=0.1)
    @validation_fraction.validator
    def _check_validation_fraction(self, _, value):
        if not (0 < value < 1):
            raise ValueError(f"validation_fraction must be between 0 and 1, not {value}")


class XGBoostRegressor(regression_solver.AbstractRegressor):
    "This is a xgboost regressor with input and output scaling"

    def __init__(self, parameters: XGBoostParameters, skip_init:bool=False):
        "Initialize the solver."
        self.parameters: XGBoostParameters
        self.error_metrics_data: ErrorMetrics = ErrorMetrics()

        if not skip_init:
            super().__init__(parameters)

    def clone(self, parameters: XGBoostParameters, skip_init:bool=False) -> 'XGBoostRegressor':
        "Clone the regressor."
        return XGBoostRegressor(parameters, skip_init=skip_init)

    def _get_scalers(self) -> tuple[LogComboScaler, LogComboScaler]:
        "Return the input and output scalers, Use StandardScaler for GPR"
        return LogComboScaler(MinMaxScaler()), LogComboScaler(MinMaxScaler())

    def _get_regressor(self) -> xgb.XGBRegressor:
        "Return the regressor."
        return xgb.XGBRegressor(objective=self.parameters.objective,
                                n_estimators=self.parameters.n_estimators,
                                max_depth=self.parameters.max_depth,
                                max_leaves=self.parameters.max_leaves,
                                max_bin=self.parameters.max_bin,
                                grow_policy=self.parameters.grow_policy,
                                learning_rate=self.parameters.learning_rate,
                                booster=self.parameters.booster,
                                verbosity=self.parameters.verbosity,
                                tree_method=self.parameters.tree_method,
                                n_jobs=self.parameters.n_jobs,
                                gamma=self.parameters.gamma,
                                min_child_weight=self.parameters.min_child_weight,
                                max_delta_step=self.parameters.max_delta_step,
                                subsample=self.parameters.subsample,
                                colsample_bytree=self.parameters.colsample_bytree,
                                colsample_bylevel=self.parameters.colsample_bylevel,
                                colsample_bynode=self.parameters.colsample_bynode,
                                reg_alpha=self.parameters.reg_alpha,
                                reg_lambda=self.parameters.reg_lambda,
                                scale_pos_weight=self.parameters.scale_pos_weight,
                                base_score=self.parameters.base_score,
                                random_state=self.parameters.random_state,
                                num_parallel_tree=self.parameters.num_parallel_tree,
                                early_stopping_rounds = self.parameters.early_stopping_rounds,
                                eval_metric = self.parameters.eval_metric)


    def save(self, path_dir: Path) -> None:
        "save the regressor, input scaler and output scaler"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted yet")

        self.regressor.save_model(path_dir / 'regressor.ubj')

        self.error_metrics_data.save(path_dir)
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

        instance.regressor.load_model(path_dir / 'regressor.ubj')


    def _fit(self, additional_callbacks:list[Callable]|None=None):
        "fit the regressor"
        scaled_split = self.data_regression.scaled_split

        if additional_callbacks is not None:
            callbacks = additional_callbacks
        else:
            callbacks = []

        self.regressor.set_params(callbacks=callbacks)

        self.regressor.fit(scaled_split.train_input, scaled_split.train_output,
                           eval_set=[(scaled_split.validation_input, scaled_split.validation_output)],
                           verbose=self.parameters.fit_verbosity)

        self.regressor.set_params(callbacks=None)
        self.fitted = True
        self.update_error_metrics()


    def _predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "predict the output data"
        input_data_scaled = self.data_regression.input_scaler.transform(input_data)

        output_data_scaled = self.regressor.predict(input_data_scaled)
        output_data_scaled = pd.DataFrame(output_data_scaled,
                                          columns=self.parameters.output_data.columns)

        # convert data types of output_data (xgboost uses float32 internally)
        for col, dtype in self.parameters.output_data.dtypes.items():
            output_data_scaled[col] = output_data_scaled[col].astype(dtype)

        output_data = self.data_regression.output_scaler.inverse_transform(output_data_scaled)

        return output_data

    def clone_tune(self, trial:optuna.trial.Trial) -> 'XGBoostRegressor':
        "Create a new regressor with the same parameters and scalers as the original."

        booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
        gamma = trial.suggest_float('gamma', 1e-8, 1e-2, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        min_child_weight = trial.suggest_float('min_child_weight', 1e-6, 1e2, log=True)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        num_parallel_tree = trial.suggest_int('num_parallel_tree', 1, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0, log=True)

        reg_data = attrs.evolve(self.parameters,
                                booster=booster,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                gamma=gamma,
                                min_child_weight=min_child_weight,
                                num_parallel_tree=num_parallel_tree,
                                learning_rate=learning_rate)

        reg = self.clone(parameters=reg_data, skip_init=True)
        reg.parameters = reg_data
        reg.data_regression = self.data_regression
        reg.regressor = reg._get_regressor()
        reg.fitted = False
        return reg

    def get_tuned_parameters(self) -> dict:
        "return the current values of the parameters that can be tuned"
        return {'booster':self.parameters.booster,
                'gamma':self.parameters.gamma,
                'max_depth':self.parameters.max_depth,
                'min_child_weight':self.parameters.min_child_weight,
                'n_estimators':self.parameters.n_estimators,
                'num_parallel_tree':self.parameters.num_parallel_tree,
                'learning_rate':self.parameters.learning_rate}

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
            callbacks = [optuna.integration.XGBoostPruningCallback(trial, observation_key='validation_0-rmse')]
            reg = self.clone_tune(trial)
            reg._fit(additional_callbacks=callbacks)
            return reg.error_metrics()['nrmse']

        study.optimize(objective,
                       n_trials=settings.number_of_trials)

        best_reg = self.clone_tune(study.best_trial)
        self.parameters = best_reg.parameters
        self.regressor = best_reg.regressor
        self.fit()
