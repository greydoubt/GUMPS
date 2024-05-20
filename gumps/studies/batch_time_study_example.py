# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Create an example BatchTimeSeries study. This multiples every item in the input by an array to increase the dimesnionaltiy. This is needed for testing."

import logging
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from gumps.studies.batch_time_study import AbstractBatchTimeStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class BatchTimeSeriesExample(AbstractBatchTimeStudy):
    "create an example BatchTimeSeries study"

    def __init__(self, model_variables:dict|None=None):
        if model_variables is None:
            self.model_variables = {'time_start':0, 'time_end':10, 'time_points':100}
        else:
            self.model_variables = model_variables
        self.time_series = np.linspace(self.model_variables['time_start'], self.model_variables['time_end'], self.model_variables['time_points'])

    def start(self):
        "initialize this study"

    def stop(self):
        "handle shutdown tasks"

    def run_row(self, row:dict) -> pd.DataFrame:
        "run a single row of data"
        temp = {}
        temp['time'] = self.time_series
        for key,value in row.items():
            temp[f"t{key}"] = value * self.time_series + value
        return pd.DataFrame(temp).set_index('time')

    def run(self, input_data:pd.DataFrame, processing_function:Callable|None=None) -> xr.Dataset:
        "run the batch simulation"
        rows = (row.to_dict() for idx,row in input_data.iterrows())
        results = [self.run_row(row) for row in rows]

        self.save_results(input_data, results)

        if processing_function is not None:
            return xr.concat((processing_function(result).to_xarray() for result in results), dim='index')
        else:
            return xr.concat((result.to_xarray() for result in results), dim='index')
