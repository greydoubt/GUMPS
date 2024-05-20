# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def ffill(arr):
    "this takes a value and fills it into following nan values"
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

class AbstractLossTimeSeries(metaclass=ABCMeta):
    "This is an abstract batch study class"

    @abstractmethod
    def run(self, input_data: list[pd.DataFrame]) -> pd.DataFrame:
        "calculate the loss from a list of dataframes"

class AbstractLossBatch(metaclass=ABCMeta):
    "Abstract Batch Loss class"

    @abstractmethod
    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "calculate the loss from a DataFrame with one entry per row"

class SimpleLossTimeSeries(AbstractLossTimeSeries):
    "this is a base class for most simple loss functions"

    def __init__(self, target:pd.DataFrame, weights:dict, time_name:str = "time"):
        "store the target data and weights"
        self.target = target
        self.weights = weights
        self.time_name = time_name

    @abstractmethod
    def loss_function(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.Series:
        "calculate the loss for a single point and return a series"

    def get_data(self, source:pd.DataFrame):
        """load source data based on target data"""
        headers = list(self.target)
        if self.time_name in headers:
            headers.remove(self.time_name)
            source = source.loc[source[self.time_name].isin(self.target[self.time_name]), headers]
            source = source.reset_index(drop=True)
        else:
            source = source.loc[:, headers]
            source = source.reset_index(drop=True)
        target = self.target.loc[:, headers]
        target = target.reset_index(drop=True)
        return source, target

    def check_dimensions(self, source, target):
        "check that the dimensions are the same and otherwise raise an error"
        if not (source.shape == target.shape):
            logger.error(f"source: {source.shape}:{list(source)}  target: {target.shape}:{list(target)}")
            raise ValueError(f'Data dimension mismatch Source: {source.shape} Target: {target.shape}')

    def apply_weights(self, series:pd.Series) -> pd.Series:
        "apply the weights to a series and return a DataFrame"
        data = {}
        for name, weight in self.weights.items():
            data[name] = (series[weight.keys()] * np.array(tuple(weight.values()))).sum()

        return pd.Series(data)

    def calculate_loss(self, source:pd.DataFrame) -> pd.Series:
        "calculate the loss for this dataframe"
        source, target = self.get_data(source)

        self.check_dimensions(source, target)

        loss = self.loss_function(source, target)

        if self.weights is not None:
            loss = self.apply_weights(loss)

        return loss

    def run(self, input_data: pd.DataFrame) -> pd.Series:
        "calculate the loss for each entry"
        return self.calculate_loss(input_data)

class SimpleLossBatch(AbstractLossBatch):
    "this is a base class for most simple batch loss functions"

    def __init__(self, target:pd.DataFrame, weights:dict):
        "store the target data and weights"
        self.target = target
        self.weights = weights

    @abstractmethod
    def loss_function(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.Series:
        "calculate the loss for a single point and return a series"

    def get_data(self, source:pd.DataFrame):
        """load source data based on target data"""
        headers = list(self.target)
        source = source.loc[:, headers]
        source = source.reset_index(drop=True)
        target = self.target.loc[:, headers]
        target = target.reset_index(drop=True)
        return source, target

    def check_dimensions(self, source, target):
        "check that the dimensions are the same and otherwise raise an error"
        if not (source.shape[1] == target.shape[1]):
            logger.error(f"source: {source.shape[1]}:{list(source)}  target: {target.shape[1]}:{list(target)}")
            raise ValueError(f'Data dimension mismatch Source: {source.shape[1]} Target: {target.shape[1]}')

    def apply_weights(self, df:pd.DataFrame) -> pd.DataFrame:
        "apply the weights to a series and return a DataFrame"
        data = {}
        for name, weight in self.weights.items():
            data[name] = (df[weight.keys()] * np.array(tuple(weight.values()))).sum(axis=1)

        return pd.DataFrame(data)

    def calculate_loss(self, source:pd.DataFrame) -> pd.DataFrame:
        "calculate the loss for this dataframe"
        source, target = self.get_data(source)

        self.check_dimensions(source, target)

        loss = self.loss_function(source, target)

        if self.weights is not None:
            loss = self.apply_weights(loss)

        return loss

    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        "calculate the loss for each entry"
        return pd.DataFrame(self.calculate_loss(input_data))

class SumSquaredErrorBatch(SimpleLossBatch):
    "calculate the loss for sum of squared error"

    def loss_function(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.Series:
        "calculate the loss for a single point and return a series"
        "sum is not actually needed for the batch process because each row is a separate record and rows are not combined for now"
        loss = pd.DataFrame((source.values - target.values)**2, columns=list(target))
        return  loss

class SumSquaredErrorTimeSeries(SimpleLossTimeSeries):
    "calculate the loss for sum of squared error"

    def loss_function(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.Series:
        "calculate the loss for a single point and return a series"
        sse = ((source - target)**2).sum()
        return  sse

class NormalizedSumSquaredErrorTimeSeries(SimpleLossTimeSeries):
    "calculate the loss for normalized sum of squared error"

    def loss_function(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.Series:
        "calculate the loss for a single point and return a series"
        norm_factor = 1.0/target.max()
        nsse = ((norm_factor*(source - target))**2).sum()
        return  nsse

class NormalizedRootMeanSquaredErrorTimeSeries(SimpleLossTimeSeries):
    "calculate the loss for normalized root mean squared error"

    def loss_function(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.Series:
        "calculate the loss for a single point and return a series"
        norm_factor = 1.0/target.max()
        sse = ((norm_factor*(source - target))**2).sum()
        mse = sse/target.count()
        rmse = np.sqrt(mse)
        return rmse

class MeanAbsoluteStandardErrorTimeSeries(SimpleLossTimeSeries):
    "calculate mean absolute standard error"

    def loss_function(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.Series:
        "calculate mean absolute standard error"
        err = (source - target).abs()
        err_sum = err.sum()
        err_n = err.count()
        num = err_sum/err_n

        val = target.apply(ffill, raw=True).fillna(0)
        val_sum = val.diff().abs().sum()
        val_n = target.count()
        den = val_sum/val_n

        mase = num/den
        return mase
