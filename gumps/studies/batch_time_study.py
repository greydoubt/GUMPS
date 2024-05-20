# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Create a batch study for time-series data. This adds an additional dimension and uses xarray instead of pandas."

import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Iterator
import functools

import pandas as pd
import xarray as xr

from gumps.common.parallel import Parallel
from gumps.studies.study import AbstractStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

process_type = Callable[[pd.DataFrame|None], pd.DataFrame|None]
runner_type = Callable[[dict], pd.DataFrame]
exception_type = Callable[[runner_type, dict, Exception], Exception|None]

class AbstractBatchTimeStudy(metaclass=ABCMeta):
    "This is an abstract batch time study class"
    input_data: pd.DataFrame|None = None
    results: xr.Dataset|None = None

    def __enter__(self):
        "handle the context manager enter event"
        self.start()
        return self

    def __exit__(self, *exc):
        "handle the context manager exit event"
        self.stop()

    @abstractmethod
    def start(self):
        "initialization"

    @abstractmethod
    def stop(self):
        "teardown"

    def iter_run(self, input_data: pd.DataFrame, processing_function:process_type) -> Iterator[tuple[int, dict, pd.DataFrame]]:
        "iterable runner, this is the default implementation"
        output_data = self.run(input_data, processing_function)

        for idx, row in input_data.iterrows():
            data = output_data.isel(index=idx).to_dataframe()
            yield idx, row.to_dict(), data

    @abstractmethod
    def run(self, input_data: pd.DataFrame, processing_function:Callable) -> xr.Dataset:
        "run the inputs and return the results with one dataframe per input"

    def full_results(self) -> tuple[pd.DataFrame, xr.Dataset]:
        "return the full results of this simulation using whatever the internal representation is for the simulation"
        if self.input_data is None or self.results is None:
            raise RuntimeError("No results to return")
        return self.input_data, self.results

    def save_results(self, input_data, results):
        "save the input data and results"
        self.input_data = input_data
        self.results = results


def batch_runner(data:dict, run_function:runner_type, processing_function:process_type, exception_handler:exception_type|None) -> tuple[dict, pd.DataFrame|None]:
    "run the simulation"
    try:
        return data, processing_function(run_function(data))
    except Exception as e:
        if exception_handler is None:
            raise e
        ret = exception_handler(run_function, data, e)

        #if the exception handler returns an exception, raise it
        #this allows the handler to handle only some exceptions
        if isinstance(ret, Exception):
            raise ret
        else:
            return data, processing_function(ret)

class BatchTimeStudyMultiProcess(AbstractBatchTimeStudy):
    "create a batch study by using a AbstractStudy and pool processing"

    def __init__(self, *, study:AbstractStudy, parallel:Parallel,
                 progress_logging:bool=False, progress_interval:int=100,
                 exception_handler: Callable | None = None):
        "create a simulation study with a parallel pool"
        self.study = study
        self.parallel = parallel
        self.progress_logging = progress_logging
        self.progress_interval = progress_interval
        self.exception_handler = exception_handler

    def start(self):
        "handle any initialization tasks that are needed"
        self.parallel.start()

    def stop(self):
        "handle any shutdown tasks that are needed"
        self.parallel.stop()

    def iter_run(self, input_data: pd.DataFrame, processing_function:process_type) -> Iterator[tuple[int, dict, pd.DataFrame]]:
        "run the inputs and return the results with one dataframe per input"
        max_rows = len(input_data)
        rows = (row.to_dict() for idx,row in input_data.iterrows())
        runner = functools.partial(batch_runner, run_function=self.study.run_data, processing_function=processing_function, exception_handler=self.exception_handler)
        for idx, (row, result) in enumerate(self.parallel.runner(runner, rows)):

            if self.progress_logging and (idx % self.progress_interval == 0 or idx == max_rows-1):
                logger.info("Batch study progress: %d/%d = %.2f%%", idx+1, max_rows, (idx+1)/max_rows*100)

            if result is not None:
                yield idx, row, result

    def run(self, input_data: pd.DataFrame, processing_function:process_type) -> xr.Dataset:
        "run the batch simulation"
        records: list[pd.DataFrame] = []
        temp_input: list[dict] = []

        for idx, row, result in self.iter_run(input_data, processing_function):
            if result is not None:
                records.append(result)
                temp_input.append(row)

        if records:
            results = xr.concat((record.to_xarray() for record in records), dim='index')
        else:
            results = xr.Dataset()

        input_data = pd.DataFrame(temp_input)

        self.save_results(input_data, results)

        return results
