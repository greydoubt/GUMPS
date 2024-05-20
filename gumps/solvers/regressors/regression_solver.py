# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract Regressor class that integrates regressor and scaler"""

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from abc import ABCMeta, abstractmethod
from typing import Callable, Iterator
from typing_extensions import Self
from pathlib import Path
import warnings

import attrs
import joblib
import pandas as pd
import numpy as np
import sklearn.metrics
import sklearn.preprocessing
import sklearn.exceptions
import optuna

from gumps.scalers.minmax_scaler import MinMaxScaler
from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.solvers.regressors.error_metrics import ErrorMetrics
from gumps.solvers.regressors.regressor_data import DataSettings, DataRegression
from gumps.scalers.scaler import AbstractScaler

@attrs.define
class OptunaParameters(object):
    "parameters for optuna study"
    number_of_trials:int = 100
    min_epochs: int = 50
    storage: str|None = None


process_type = Callable[[pd.DataFrame], pd.DataFrame]

def load_scaler(path_dir: Path) -> AbstractScaler:
    try:
        return joblib.load(path_dir)
    except (UserWarning, sklearn.exceptions.InconsistentVersionWarning) as exc:
        raise RuntimeError("Failed to load input scaler") from exc

@attrs.define(kw_only=True)
class RegressionParameters(DataSettings):
    "Regression parameters"

class AbstractRegressor(metaclass=ABCMeta):
    "Abstract regressor class"
    input_data: pd.DataFrame|None = None
    results: list[pd.DataFrame]|None = None
    data_regression: DataRegression|None = None
    input_scaler = None
    output_scaler = None

    #These are here to allow older regressors to be loaded and converted to the new format
    train_indices = None
    validation_indices = None

    def __init__(self, parameters: RegressionParameters) -> None:
        "Initialize the regressor"
        self.parameters = parameters
        self.regressor = self._get_regressor()
        input_scaler, output_scaler = self._get_scalers()
        self.data_regression: DataRegression|None = DataRegression(settings=self.parameters,
                                                              input_scaler=input_scaler,
                                                              output_scaler=output_scaler)
        self.fitted: bool = False
        self.error_metrics_data: ErrorMetrics = ErrorMetrics()

    @abstractmethod
    def clone(self, parameters: RegressionParameters) -> Self:
        "Clone the regressor"

    def _get_scalers(self) -> tuple[LogComboScaler, LogComboScaler]:
        "Return the input and output scalers, Use MinMaxScaler for regression"
        return LogComboScaler(MinMaxScaler()), LogComboScaler(MinMaxScaler())


    #Implement the Batch Interface
    def __enter__(self):
        "handle the context manager enter event"
        if not self.fitted:
            raise RuntimeError("Regressor must be fitted before it can be entered")


    def __exit__(self, *exc):
        "handle the context manager exit event"
        ...


    def start(self):
        "initialization"
        if not self.fitted:
            raise RuntimeError("Regressor must be fitted before it can be started")


    def stop(self):
        "teardown"
        ...


    def run(self, input_data: pd.DataFrame, processing_function:process_type) -> pd.DataFrame:
        "run the inputs and return the results with one dataframe per input"
        results = self.predict(input_data)
        self.save_results(input_data, results)
        return processing_function(results)


    def iter_run(self, input_data: pd.DataFrame, processing_function:process_type) -> Iterator[tuple[int, dict, pd.Series|dict]]:
        "iterable runner, this is the default implementation"
        output_data = self.run(input_data, processing_function)

        for idx, ((_, row), (_, data)) in enumerate(zip(input_data.iterrows(), output_data.iterrows())):
            data.name = None
            yield idx, row.to_dict(), data


    def full_results(self):
        "return the full results of this simulation using whatever the internal representation is for the simulation"
        return self.input_data, self.results


    def save_results(self, input_data, results):
        "save the input data and results"
        self.input_data = input_data
        self.results = results


    @abstractmethod
    def _get_regressor(self):
        "Return the regressor."

    @abstractmethod
    def get_tuned_parameters(self) -> dict:
        "return the current value of parameters that can be tuned"

    @abstractmethod
    def auto_tune(self, settings:OptunaParameters) -> None:
        "Automatically tune the regressor."


    def save(self, path_dir: Path) -> None:
        "save the regressor, input scaler and output scaler"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted yet")

        self.error_metrics_data.save(path_dir)
        joblib.dump(self.regressor, path_dir / "regressor.joblib")
        joblib.dump(self.parameters, path_dir / "parameters.joblib")
        with open(path_dir / 'regressor.txt', 'w', encoding='utf-8') as f:
            f.write(self.__class__.__name__)

        self.data_regression.save(path_dir)


    @classmethod
    def _load_instance(cls, path_dir:Path, instance):
        """load method that a child-class can use to load additional components
        If the components fail to load an RuntimeError must be raised
        """


    @classmethod
    def _load_regressor(cls, instance, path_dir:Path) -> None:
        "load the regressor and raise an exception if it fails to load"
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=sklearn.exceptions.InconsistentVersionWarning)
            warnings.simplefilter("error", UserWarning)
            try:
                instance.regressor = joblib.load(path_dir / "regressor.joblib")
            except (UserWarning, sklearn.exceptions.InconsistentVersionWarning) as exc:
                raise RuntimeError("Failed to load regressor") from exc


    @classmethod
    def load_instance(cls, path_dir:Path):
        "load the regressor and raise an exception if it fails to load"
        instance = cls.__new__(cls)
        instance.fitted = True

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=sklearn.exceptions.InconsistentVersionWarning)
            warnings.simplefilter("error", UserWarning)

            cls._load_regressor(instance, path_dir)

            if not DataRegression.has_data(path_dir):
                instance.input_scaler = load_scaler(path_dir / "input_scaler.joblib")
                instance.output_scaler = load_scaler(path_dir / "output_scaler.joblib")
            cls._load_instance(path_dir, instance)

        return instance

    def load_error_metrics_data(self, path_dir:Path):
        "load the error metrics data"
        self.error_metrics_data = ErrorMetrics()
        try:
            self.error_metrics_data.load(path_dir)
        except FileNotFoundError:
            self.update_error_metrics()

    def update_load_indices(self):
        "update the indices after loading if they are defined, this handles older objects"
        if self.train_indices is not None:
            self.parameters.train_indices = self.train_indices

        if self.validation_indices is not None:
            self.parameters.validation_indices = self.validation_indices

    @classmethod
    def load(cls, path_dir: Path, auto_rebuild: bool = True, auto_resave: bool = True):
        """
        Load the regressor, input scaler, and output scaler.

        Parameters:
        - auto_rebuild: If True, rebuild the regressor if it fails to load.
                        If False and the regressor fails to load or raises a warning, an exception will be raised.
        - auto_resave: If True, resave the regressor if it was rebuilt.
        """
        try:
            instance = cls.load_instance(path_dir)

            try:
                #try to load the parameters from the instance
                instance.parameters
            except AttributeError:
                instance.parameters = cls.load_parameters(path_dir)

            instance.update_load_indices()

        except RuntimeError as exc:
            if auto_rebuild:
                instance = cls.rebuild_model(path_dir, auto_resave=auto_resave)
            else:
                raise RuntimeError("Regressor failed to load.") from exc

        if instance.data_regression is None:
            if DataRegression.has_data(path_dir):
                instance.data_regression = DataRegression.load(path_dir)
            elif instance.input_scaler is not None and instance.output_scaler is not None:
                instance.data_regression = DataRegression(settings=instance.parameters,
                                                        input_scaler=instance.input_scaler,
                                                        output_scaler=instance.output_scaler)
        instance.load_error_metrics_data(path_dir)
        return instance

    @classmethod
    def load_parameters(cls, path_dir: Path):
        "create new parameters for the regressor"
        parameters = joblib.load(path_dir / "parameters.joblib")
        data = {}
        for field in attrs.fields(parameters.__class__):
            data[field.name] = getattr(parameters, field.name, field.default)
        return parameters.__class__(**data)

    @classmethod
    def rebuild_model(cls, path_dir: Path, auto_resave: bool = True):
        "Rebuild the regressor and optionally resave it"
        logger.warning("Regressor failed to load, rebuilding regressor.")
        parameters = cls.load_parameters(path_dir)
        instance = cls(parameters)
        instance.fit()

        if auto_resave:
            logger.warning("Regressor was rebuilt, resaving regressor.")
            instance.save(path_dir)

        return instance


    def fit(self):
        "fit the regressor"
        self._fit()


    def update_error_metrics(self):
        "update the error metrics"
        self.error_metrics_data.set_data(self.parameters.output_data, self.predict(self.parameters.input_data))


    @abstractmethod
    def _fit(self):
        "fit the regressor"


    def error_metrics(self, multioutput:str="uniform_average") -> pd.Series:
        "return the score of the regressor"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted.")
        return self.error_metrics_data.get_metrics(multioutput)

    def error_frame(self) -> pd.DataFrame:
        "return the error metrics frame"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted.")
        return self.error_metrics_data.get_metrics_frame()

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "predict the output data"
        if not self.fitted:
            raise RuntimeError("Regressor has not been fitted.")
        return self._predict(input_data)


    def _predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "predict the output data"
        input_data_scaled = self.data_regression.input_scaler.transform(input_data)

        output_data_scaled = self.regressor.predict(input_data_scaled)
        output_data_scaled = pd.DataFrame(output_data_scaled,
                                          columns=self.parameters.output_data.columns)

        output_data = self.data_regression.output_scaler.inverse_transform(output_data_scaled)

        return output_data

    def update_data(self, input_data:pd.DataFrame, output_data:pd.DataFrame):
        "update the input and output data for a regressor and refit the regressor"
        self.data_regression.update_data(input_data, output_data)
        self.fit()
