# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#Create an EnsembleRegressor class that inherits from the Regressor class and
#uses ShuffleSplit to split the data, create a list of regressors and use an ensemble to combine them

import logging
import optuna

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from pathlib import Path

import attrs
import joblib
import pandas as pd
import sklearn.preprocessing
import sklearn.exceptions
import sklearn.model_selection
import numpy as np

import gumps.solvers.regressors.regression_solver as regression_solver
from gumps.scalers.scaler import AbstractScaler
from gumps.solvers.regressors.error_metrics import ErrorMetrics
from gumps.solvers.regressors.regressor_data import Split, keep_unique

#DO NOT CHANGE THIS IMPORT STATEMENT. It is a circular import that is necessary for the regression_loader to work
import gumps.solvers.regressors.regression_loader as regression_loader

@attrs.define
class Uncertainty:
    prediction: pd.DataFrame
    standard_deviation: pd.DataFrame
    minimum: pd.DataFrame
    maximum: pd.DataFrame

@attrs.define
class EnsembleParameters(regression_solver.RegressionParameters):
    "Ensemble regression parameters with validation on the number of regressors"
    regressor: regression_solver.AbstractRegressor
    number_regressors: int = attrs.field()
    validation_fraction: float = attrs.field(default=0.1)
    random_state: int | None = None
    weight_ensemble: bool = True
    weight_tolerance: float = 1e-5 #if all the nrmse weights are below this tolerance, then the weights are equal for a response variable

    @number_regressors.validator
    def check_number_regressors(self, attribute, value):
        "validate the number of regressors"
        if not (value > 0 and value == int(value)):
            raise ValueError(f"Only integers greater than 0 are allowed for Ensemble regression {attribute} = {value}")

    @validation_fraction.validator
    def check_validation_fraction(self, attribute, value):
        "validate the validation fraction"
        if not (0 < value < 1):
            raise ValueError(f"Only values between 0 and 1 are allowed for Ensemble regression {attribute} = {value}")

class EnsembleRegressor(regression_solver.AbstractRegressor):
    "Implement an ensemble regressor that uses ShuffleSplit to split the data"

    def __init__(self, parameters: EnsembleParameters):
        self.parameters: EnsembleParameters
        super().__init__(parameters)
        self.splits: list[Split] = self._create_splits()
        self.regressors: list[regression_solver.AbstractRegressor] = self._create_regressor()
        self.weights: pd.DataFrame|None = None
        self.error_metrics_data: ErrorMetrics = ErrorMetrics()


    def _get_scalers(self) -> tuple[AbstractScaler, AbstractScaler]:
        "Return the scalers (don't do any scaling for an ensemble regressor, each regressor will do its own scaling)"
        return self.regressor._get_scalers()


    def _get_regressor(self) -> regression_solver.AbstractRegressor:
        "Return the regressor"
        return self.parameters.regressor


    def _create_splits(self) -> list[Split]:
        "Create the splits"
        split = sklearn.model_selection.ShuffleSplit(n_splits=self.parameters.number_regressors,
                                                     test_size=self.parameters.validation_fraction,
                                                     random_state=self.parameters.random_state)
        splits = []
        for train_index, validation_index in split.split(self.parameters.input_data):
            train_input = self.parameters.input_data.iloc[train_index]
            validation_input = self.parameters.input_data.iloc[validation_index]
            train_output = self.parameters.output_data.iloc[train_index]
            validation_output = self.parameters.output_data.iloc[validation_index]

            splits.append(Split(train_input=train_input,
                                validation_input=validation_input,
                                train_output = train_output,
                                validation_output=validation_output,
                                full_input = self.parameters.input_data,
                                full_output = self.parameters.output_data))
        return splits


    def _create_regressor(self) -> list[regression_solver.AbstractRegressor]:
        "Create the regressors"
        regressors = []
        for split in self.splits:
            parameters = attrs.evolve(self.parameters.regressor.parameters,
                                      input_data=split.full_input,
                                      output_data=split.full_output,
                                      train_test_split="manual",
                                      train_indices=list(split.train_input.index),
                                      validation_indices=list(split.validation_input.index),)

            regressor = self.parameters.regressor.clone(parameters)
            regressors.append(regressor)
        return regressors


    def _fit(self):
        "Fit the regressors"

        #The base regressor needs to be fitted so that it can be saved and loaded and later used for tuning
        self.regressor.fit()

        for regressor in self.regressors:
            regressor.fit()
        self.weights = self._create_weights()
        self.fitted = True
        self.update_error_metrics()


    def _create_error(self) -> pd.DataFrame:
        "Create the error"
        errors = {}

        for idx, regressor in enumerate(self.regressors):
            errors[idx] = regressor.error_frame().loc['nrmse']

        error_df = pd.DataFrame(errors).T
        return error_df

    def _create_weights(self) -> pd.DataFrame:
        "Create the weights"
        if self.parameters.weight_ensemble:
            return self._create_weights_error()
        else:
            return self._create_weights_uniform()

    def _create_weights_error(self) -> pd.DataFrame:
        error_df = self._create_error()
        max_error = error_df.max()
        below_tolerance = max_error < self.parameters.weight_tolerance
        error_df.loc[:, below_tolerance] = 1
        weights = (1.0/error_df)/(1.0/error_df).sum()
        return weights

    def _create_weights_uniform(self) -> pd.DataFrame:
        "Create the weights uniformly"
        row_number = self.parameters.number_regressors
        column_names = self.parameters.output_data.columns
        weights = pd.DataFrame(1/row_number, index=range(row_number), columns=column_names)
        return weights


    def clone(self, parameters: EnsembleParameters) -> 'EnsembleRegressor':
        "Clone the regressor"
        return EnsembleRegressor(parameters)

    def _predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "Predict the output"
        predictions = []
        for idx, regressor in enumerate(self.regressors):
            prediction = regressor.predict(input_data)
            predictions.append(prediction * self.weights.loc[idx])

        array_weight_3d = np.stack([df.to_numpy() for df in predictions], axis=-1)
        mean_weight_df = pd.DataFrame(np.sum(array_weight_3d, axis=2), index=predictions[0].index, columns=predictions[0].columns)
        return mean_weight_df

    def predict_with_uncertainty(self, input_data: pd.DataFrame) -> Uncertainty:
        "Predict the output with uncertainty"
        weighted_predictions = []
        predictions = []

        for idx, regressor in enumerate(self.regressors):
            prediction = regressor.predict(input_data)
            weighted_predictions.append(prediction * self.weights.loc[idx])
            predictions.append(prediction)


        array_weight_3d = np.stack([df.to_numpy() for df in weighted_predictions], axis=-1)
        mean_weight_df = pd.DataFrame(np.sum(array_weight_3d, axis=2), index=weighted_predictions[0].index, columns=weighted_predictions[0].columns)

        array_3d = np.stack([df.to_numpy() for df in predictions], axis=-1)
        min_df = pd.DataFrame(np.min(array_3d, axis=2), index=predictions[0].index, columns=predictions[0].columns)
        max_df = pd.DataFrame(np.max(array_3d, axis=2), index=predictions[0].index, columns=predictions[0].columns)
        std_df = pd.DataFrame(np.std(array_3d, axis=2), index=predictions[0].index, columns=predictions[0].columns)

        return Uncertainty(prediction=mean_weight_df,
                           standard_deviation=std_df,
                           minimum=min_df,
                           maximum=max_df)


    def update_data(self, input_data:pd.DataFrame, output_data:pd.DataFrame):
        "update the data"
        new_input = pd.concat([self.parameters.input_data, input_data])
        new_output = pd.concat([self.parameters.output_data, output_data])
        new_input, new_output = keep_unique(new_input, new_output)
        self.parameters.input_data = new_input
        self.parameters.output_data = new_output

        self.splits = self._create_splits()
        for split,regressor in zip(self.splits, self.regressors):
            regressor.data_regression.reset_split(split)
        self.fit()


    def save(self, path_dir: Path) -> None:
        "save the regressor, input scaler and output scaler"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted yet")

        self.error_metrics_data.save(path_dir)

        regressor_dir = path_dir / "regressor"
        regressor_dir.mkdir(parents=True, exist_ok=True)
        self.regressor.save(regressor_dir)

        joblib.dump(self.parameters, path_dir / "parameters.joblib")
        with open(path_dir / 'regressor.txt', 'w', encoding='utf-8') as f:
            f.write(self.__class__.__name__)

        joblib.dump(self.weights, path_dir / "weights.joblib")
        joblib.dump(self.splits, path_dir / "splits.joblib")

        for idx, regressor in enumerate(self.regressors):
            regressor_dir = path_dir / "regressors" / f"{idx}"
            regressor_dir.mkdir(parents=True, exist_ok=True)
            regressor.save(regressor_dir)

        self.data_regression.save(path_dir)

    @classmethod
    def load(cls, path_dir:Path, auto_rebuild:bool = True, auto_resave:bool = True):
        "load the regressor"
        parameters = joblib.load(path_dir / "parameters.joblib")
        regressor = regression_loader.load_regressor(path_dir / "regressor",
                                                     auto_rebuild=auto_rebuild,
                                                     auto_resave=auto_resave)

        weights = joblib.load(path_dir / "weights.joblib")
        splits = joblib.load(path_dir / "splits.joblib")

        regressors = []
        for idx in range(parameters.number_regressors):
            regressor = regression_loader.load_regressor(path_dir / "regressors" / f"{idx}",
                                                         auto_rebuild=auto_rebuild,
                                                         auto_resave=auto_resave)
            regressors.append(regressor)

        ensemble_regressor = cls.__new__(cls)
        ensemble_regressor.parameters = parameters
        ensemble_regressor.regressor = regressor
        ensemble_regressor.weights = weights
        ensemble_regressor.splits = splits
        ensemble_regressor.regressors = regressors
        ensemble_regressor.fitted = True
        ensemble_regressor.load_error_metrics_data(path_dir)

        return ensemble_regressor

    def auto_tune(self, *args, **kw) -> None:
        "Automatically tune the regressor."
        self.regressor.auto_tune(*args, **kw)
        self.regressors = self._create_regressor()
        self.fit()

    def get_tuned_parameters(self) -> dict:
        "Get the tuned parameters"
        return self.regressor.get_tuned_parameters()

