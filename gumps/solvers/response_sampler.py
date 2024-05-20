# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#This module creates a response sampler. It is used to visualize the response of a study
#in order to better understand it before attempting regression models, parameter estimation, or optimization

import logging
import itertools

import attrs
import numpy as np
import pandas as pd

from gumps.solvers.batch_solver import AbstractBatchSolver

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class ResponseSamplerParameters:
    "Interface for ResponseSampler required settings"
    lower_bound: pd.Series = attrs.field(converter=pd.Series)
    upper_bound: pd.Series = attrs.field(converter=pd.Series)
    baseline: pd.Series =  attrs.field(converter=pd.Series)
    points_1d: int | None = None
    points_2d_per_dimension: int | None = None


def check_response(func):
    def wrapper(self, *args, **kwargs):
        if self.response is None:
            raise RuntimeError("tell must be called before split_response")

        if self.request is None:
            raise RuntimeError("ask must be called before split_response")

        return func(self, *args, **kwargs)

    return wrapper

class ResponseSampler(AbstractBatchSolver):


    def __init__(self, *, solver_settings: ResponseSamplerParameters):
        "initialize the ResponseSamplerParameters"
        self.solver_settings = solver_settings
        self.response: pd.DataFrame | None = None
        self.request: pd.DataFrame | None = None


    def has_next(self) -> bool:
        return self.response is None


    def generate_1d(self) -> pd.DataFrame:
        "Generate 1d points"
        frames = []
        for key in self.solver_settings.baseline.keys():
            frames.append(self.generate_1d_keys(key))
        return pd.concat(frames)


    def generate_1d_keys(self, key: str) -> pd.DataFrame:
        "Generate 1d points for a single key"
        df = pd.DataFrame()
        df[key] = np.linspace(self.solver_settings.lower_bound[key], self.solver_settings.upper_bound[key], self.solver_settings.points_1d)

        for name in self.solver_settings.baseline.keys():
            if name != key:
                df[name] = self.solver_settings.baseline[name]
        return df


    def generate_2d(self) -> pd.DataFrame:
        "generate 2d variations by changing two variables at a time"
        frames = []
        for key1, key2 in itertools.combinations(self.solver_settings.baseline.keys(), 2):
            frames.append(self.generate_2d_keys(key1, key2))
        return pd.concat(frames)


    def generate_2d_keys(self, key1:str, key2:str) -> pd.DataFrame:
        "generate 2d variations for a single key"
        var1 = np.linspace(self.solver_settings.lower_bound[key1], self.solver_settings.upper_bound[key1], self.solver_settings.points_2d_per_dimension)
        var2 = np.linspace(self.solver_settings.lower_bound[key2], self.solver_settings.upper_bound[key2], self.solver_settings.points_2d_per_dimension)
        vals = np.array(np.meshgrid(var1, var2)).T.reshape(-1,2)
        df = pd.DataFrame(vals, columns=[key1, key2])
        for name in self.solver_settings.baseline.keys():
            if name != key1 and name != key2:
                df[name] = self.solver_settings.baseline[name]
        return df


    def ask(self) -> pd.DataFrame:
        "Return the array for sampling"
        if self.solver_settings.points_1d is None and self.solver_settings.points_2d_per_dimension is None:
            raise ValueError("At least one of points_1d or points_2d must be set")

        baseline = self.solver_settings.baseline.to_frame().T

        frames = [baseline]

        if self.solver_settings.points_1d is not None:
            frames.append(self.generate_1d())

        if self.solver_settings.points_2d_per_dimension is not None:
            frames.append(self.generate_2d())

        request = pd.concat(frames, ignore_index=True)

        #depending on the exact values chosen there may be duplicates
        request = request.drop_duplicates().reset_index(drop=True)

        self.request = request

        return request


    def tell(self, response: pd.DataFrame, request:pd.DataFrame|None=None) -> None:
        "provide the response"
        if self.request is None:
            raise RuntimeError("ask must be called before tell")

        if request is None:
            request = self.request

        if response.shape[0] != request.shape[0]:
            raise ValueError("The response and request must have the same number of rows")
        self.request = request
        self.response = response


    def get_response(self) -> pd.DataFrame:
        "Return the response"
        if self.response is None:
            raise RuntimeError("tell must be called before get_response")
        return self.response


    @check_response
    def get_baseline_response(self) -> pd.DataFrame:
        "get the baseline results which is always the first simulation"
        baseline = pd.concat([self.request.loc[0].to_frame(), self.response.loc[0].to_frame()]).T

        #remove the name (index = 0)
        baseline.name = None

        return baseline


    @check_response
    def split_response(self)-> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        "split the response into multiple dataframes, raise an exception if the response is not set"
        baseline = self.get_baseline_response()

        data_1d = self.get_response_1d()

        data_2d = self.get_response_2d()

        return baseline, data_1d, data_2d


    @check_response
    def get_response_1d(self) -> dict[str, pd.DataFrame]:
        "return the response split into 1d dataframes"
        data_1d = {}

        baseline = self.solver_settings.baseline

        for key in baseline.keys():
            #find the indexes where all values except the key is equal to the mean value in the input_df

            keys_used = self.request.columns.difference([key])

            indexes = self.request[keys_used].eq(baseline[keys_used]).all(axis=1)

            #join the input and output dataframes on the indexes
            data = self.request.loc[indexes].join(self.response[indexes])

            data_1d[key] = data

        return data_1d


    @check_response
    def get_response_2d(self) -> dict[str, pd.DataFrame]:
        "return the response split into 2d dataframes"
        data_2d = {}

        baseline = self.solver_settings.baseline

        for key1, key2 in itertools.combinations(baseline.keys(), 2):
            #find the indexes where all values except the key is equal to the mean value in the input_df

            keys_used = self.request.columns.difference([key1, key2])

            indexes = self.request[keys_used].eq(baseline[keys_used]).all(axis=1)

            #join the input and output dataframes on the indexes
            data = self.request.loc[indexes, [key1, key2]].join(self.response[indexes])

            data_2d[f'{key1}___{key2}'] = data

        return data_2d
