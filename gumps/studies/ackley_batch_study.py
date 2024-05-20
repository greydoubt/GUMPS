# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Callable

import numpy as np
import pandas as pd

from gumps.studies.batch_study import AbstractBatchStudy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class AckleyBatchStudy(AbstractBatchStudy):
    "Ackley Batch study"

    def __init__(self, model_variables:dict):
        self.model_variables = model_variables

    def start(self):
        "initialize this study"

    def stop(self):
        "handle shutdown tasks"

    def run(self, input_data:pd.DataFrame, processing_function:Callable) -> pd.DataFrame:
        "run the batch simulation"
        sum_squared = (input_data['x']**2).apply(sum)
        len_x = input_data['x'].apply(len)
        term1 =(-input_data['a']) * np.exp( - input_data['b'] * np.sqrt(sum_squared/len_x))
        prod = (input_data['c']*input_data['x'])
        mean = [np.mean(np.cos(x)) for x in prod.to_list()]
        term2 = - np.exp(mean)
        total = term1 + term2 + input_data['a'] + np.exp(1)

        output_data = pd.DataFrame({'total': total})
        self.save_results(input_data, output_data)

        return processing_function(output_data)