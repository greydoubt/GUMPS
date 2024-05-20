# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#create a parametric sweep app for time series data, each study returns a dataframe of time series
#data and then this needs to be stored to hdf5 periodically due to memory constraints

import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional, Callable
from pathlib import Path
import time
import h5py

from gumps.common import hdf5
from gumps.common.app_utils import run_batch_time_iterator
from gumps.studies.batch_time_study import AbstractBatchTimeStudy
import gumps.solvers.sampler

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class ParametricSweepTimeSeriesApp:
    "Create a parametric sweep time series app"

    def __init__(self, *, parameters: gumps.solvers.sampler.SamplerSolverParameters,
        processing_function: Callable, directory:Optional[Path]=None,
        batch: AbstractBatchTimeStudy,pre_processing_function: Callable|None=None):
        "initialize the Parametric Sweep app"
        self.parameters = parameters
        self.processing_function = processing_function
        self.pre_processing_function = pre_processing_function
        self.batch = batch

        self.directory = directory
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

        self.factors: pd.DataFrame | None = None
        self.responses: xr.Dataset | None = None

    def answer(self) -> xr.Dataset:
        "return the final result"
        if self.responses is None:
            raise RuntimeError("The parametric sweep app has not been run yet")
        return self.responses

    def save_data_hdf5(self):
        "save all the data to hdf5"
        if self.directory is None or self.factors is None or self.responses is None:
            raise RuntimeError("The parametric sweep app has not been run yet")

        data = hdf5.H5( (self.directory / "data.h5").as_posix() )
        data.root.factors = self.factors.to_numpy()
        data.root.responses = self.responses.to_array().values
        data.root.factor_names = self.factors.columns
        data.root.response_names = list(self.responses.keys())
        data.root.time_values = self.responses.time.values
        data.save()

    def update_hdf5(self, factors:pd.DataFrame, responses:list[pd.DataFrame]):
        "update the hdf5 file with the new data"
        if self.directory is None:
            raise RuntimeError("The directory must be set to run the streaming app")

        data = self.directory / "data.h5"

        if len(factors) > 0 and len(factors) == len(responses):
            if data.exists():
                with h5py.File(data, "a") as hf:
                    factor_data = factors.to_numpy()
                    hf["factors"].resize((hf["factors"].shape[0] + factor_data.shape[0]), axis = 0)
                    hf["factors"][-factor_data.shape[0]:] = factor_data

                    response_data = np.stack([record.values for record in responses], axis=0)
                    hf["responses"].resize((hf["responses"].shape[0] + response_data.shape[0]), axis = 0)
                    hf["responses"][-response_data.shape[0]:] = response_data
            else:
                with h5py.File(data, "w") as hf:
                    factor_data = factors.to_numpy()
                    hf.create_dataset("factors", data=factor_data, maxshape=(None, factor_data.shape[1]))

                    response_data = np.stack([record.values for record in responses], axis=0)
                    hf.create_dataset("responses", data=response_data, maxshape=(None, response_data.shape[1], response_data.shape[2]))

                    hf.create_dataset("factor_names", data=[x.encode() for x in factors.columns])
                    hf.create_dataset("response_names", data=[str(x).encode() for x in responses[0]])
                    hf.create_dataset("time_values", data=responses[0].index.values)
        else:
            logger.info("No new data to save")

    def run(self):
        solver = gumps.solvers.sampler.SamplerSolver(solver_settings=self.parameters)
        with self.batch:
            while solver.has_next():
                factors = solver.ask()

                factors, responses = run_batch_time_iterator(self.batch, factors, self.processing_function, self.pre_processing_function)

                solver.tell(responses)

                self.factors = factors
                self.responses = responses

    def process_interval(self, factors:pd.DataFrame, keep_indexes:list[int], records:list[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        factors_local = factors.iloc[keep_indexes].reset_index(drop=True)

        keep_idx: list[int] = []
        for idx, record in enumerate(records):
            if record.isna().any().any():
                logger.warning(f"The following inputs produced nan-values as an output and will be dropped \n{factors_local.iloc[idx]}")
            else:
                keep_idx.append(idx)

        responses = [records[idx] for idx in keep_idx]
        factors_local = factors_local.iloc[keep_idx].reset_index(drop=True)

        return factors_local, responses

    def run_streaming(self, save_interval_seconds:int=300):
        "run the simulation and save the results to hdf5 periodically"
        if self.directory is None:
            raise RuntimeError("The directory must be set to run the streaming app")

        solver = gumps.solvers.sampler.SamplerSolver(solver_settings=self.parameters)
        with self.batch:
            while solver.has_next():
                factors = solver.ask()

                if self.pre_processing_function is not None:
                    input_data_processed = self.pre_processing_function(factors)
                else:
                    input_data_processed = factors

                keep_indexes: list[int] = []
                records: list[pd.DataFrame] = []


                start_time = time.time()
                for idx, _, data in self.batch.iter_run(input_data_processed, self.processing_function):
                    keep_indexes.append(idx)
                    records.append(data)

                    if time.time() - start_time > save_interval_seconds:
                        factors_local, responses = self.process_interval(factors, keep_indexes, records)
                        self.update_hdf5(factors_local, responses)

                        start_time = time.time()
                        keep_indexes = []
                        records = []

                factors_local, responses = self.process_interval(factors, keep_indexes, records)
                self.update_hdf5(factors_local, responses)


                solver.tell(xr.Dataset())
