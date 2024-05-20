# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Create a sphere study and line study to use for testing"

import logging
from typing import Callable

import pandas as pd

from gumps.studies.batch_study import AbstractBatchStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class BatchSphereStudy(AbstractBatchStudy):
    "batch version of sphere study (designed to approximate surrogate model)"

    def __init__(self, model_variables:dict):
        self.model_variables = model_variables
        self.center = {key.replace('a', 'x'):value for key,value in self.model_variables.items()}

    def start(self):
        "initialize this study"

    def stop(self):
        "handle shutdown tasks"

    def run(self, input_data:pd.DataFrame, processing_function:Callable) -> pd.DataFrame:
        "run the batch simulation"
        diff = (input_data - self.center)**2
        mapper = {key:key.replace('x', 'd') for key in self.center}
        diff.rename(columns=mapper, inplace=True)
        diff['total'] = diff.sum(axis=1)

        self.save_results(input_data, diff)

        return processing_function(diff)

class BatchLineStudy(BatchSphereStudy):
    "create a batch line study, this study just does a diff of the variables and is useful for analytical answers for statistics problems"

    def run(self, input_data:pd.DataFrame, processing_function:Callable) -> pd.DataFrame:
        "run the batch simulation"
        diff = input_data - self.center
        mapper = {key:key.replace('x', 'd') for key in self.center}
        diff.rename(columns=mapper, inplace=True)
        diff['total'] = diff.sum(axis=1)

        self.save_results(input_data, diff)
        return processing_function(diff)
