# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#create utilities for applications
from typing import Callable
import pandas as pd
from gumps.studies.batch_study import AbstractBatchStudy
from gumps.studies.batch_time_study import AbstractBatchTimeStudy
import xarray as xr

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def run_batch_iterator(batch:AbstractBatchStudy, input_data:pd.DataFrame, processing_function:Callable, pre_processing_function:Callable|None) -> tuple[pd.DataFrame, pd.DataFrame]:
    "run the batch and return the results as an iterator"
    if pre_processing_function is not None:
        input_data_processed = pre_processing_function(input_data)
    else:
        input_data_processed = input_data

    keep_indexes: list[int] = []
    temp_output: list[dict|pd.Series] = []

    for idx, _, data in batch.iter_run(input_data_processed, processing_function):
        keep_indexes.append(idx)
        temp_output.append(data)

    output_data = pd.DataFrame(temp_output)
    input_data = input_data.iloc[keep_indexes].reset_index(drop=True)

    nan_rows = input_data[output_data.isna().any(axis=1)]
    if len(nan_rows):
        logger.warning(f"The following inputs produced nan-values as an output and will be dropped \n{nan_rows}")

    input_data = input_data.drop(index=nan_rows.index).reset_index(drop=True)
    output_data = output_data.drop(index=nan_rows.index).reset_index(drop=True)

    return input_data, output_data

def run_batch_time_iterator(batch:AbstractBatchTimeStudy, input_data:pd.DataFrame, processing_function:Callable, pre_processing_function:Callable|None) -> tuple[pd.DataFrame, xr.Dataset]:
    "run the batch and return the results as an iterator"
    if pre_processing_function is not None:
        input_data_processed = pre_processing_function(input_data)
    else:
        input_data_processed = input_data

    keep_indexes: list[int] = []
    records: list[pd.DataFrame] = []

    for idx, _, data in batch.iter_run(input_data_processed, processing_function):
        keep_indexes.append(idx)
        records.append(data)

    input_data = input_data.iloc[keep_indexes].reset_index(drop=True)
    distribution = xr.concat((record.to_xarray() for record in records), dim='index')
    distribution = distribution.assign_coords(index=input_data.index)

    distribution = distribution.dropna('index')
    if len(distribution.index) != len(input_data):
        expected = set(input_data.index)
        actual = set(distribution.index.to_numpy())
        missing = expected - actual

        input_data = input_data[input_data.index.isin(actual)]

        logger.warning(f"The following inputs produced nan-values as an output and will be dropped \n{missing}")

    input_data = input_data.reset_index(drop=True)
    distribution = distribution.drop(index=input_data.index)

    return input_data, distribution