# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create an adaptive sampler app, this app only works with an already fitted GaussianRegressor
After this method has run the regressor passed into the parameters will be updated with the
new data and fitted again.

To save the new version simple call the save method of the regressor. This process can be repeated
to further improve the regressor"""

import logging
from typing import Callable

import attrs
import pandas as pd

from gumps.solvers.adaptive_solver import (AdaptiveSamplerParameters,
                                           GaussianRegressorAdaptiveSampler)
from gumps.solvers.regressors.gaussian_regressor import GaussianRegressor
from gumps.studies.batch_study import AbstractBatchStudy
from gumps.common.app_utils import run_batch_iterator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

user_continue_type = Callable[[pd.DataFrame|None, pd.DataFrame|None], bool]

@attrs.define
class AdaptiveSamplerAppParameters:
    "Parameters for the adaptive sampler app"
    regressor: GaussianRegressor
    sampler_parameters: AdaptiveSamplerParameters
    user_continue: user_continue_type|None = None

class AdaptiveSamplerApp:
    "Create an adaptive sampler app"

    def __init__(self, *, parameters: AdaptiveSamplerAppParameters,
                 batch: AbstractBatchStudy,
                 processing_function: Callable | None,
                 pre_processing_function : Callable | None=None):
        "initialize the adaptive sampler app"
        self.parameters = parameters
        self.batch = batch
        self.processing_function = processing_function
        self.pre_processing_function = pre_processing_function
        self.new_inputs: pd.DataFrame | None = None
        self.new_outputs: pd.DataFrame | None = None

    def update_results(self, new_inputs: pd.DataFrame, new_outputs: pd.DataFrame):
        "update the results of the app"
        if self.new_inputs is None:
            self.new_inputs = new_inputs
        else:
            self.new_inputs = pd.concat([self.new_inputs, new_inputs])

        if self.new_outputs is None:
            self.new_outputs = new_outputs
        else:
            self.new_outputs = pd.concat([self.new_outputs, new_outputs])

    def run_user_continue(self):
        "run the user continue function"
        if self.parameters.user_continue is not None:
            return self.parameters.user_continue(self.new_inputs, self.new_outputs)
        return True

    def run(self):
        "run the adaptive sampler app"
        sampler = GaussianRegressorAdaptiveSampler(self.parameters.regressor,
                                                   self.parameters.sampler_parameters)

        with self.batch:
            while self.run_user_continue() and sampler.has_next():
                input_data = sampler.ask()

                input_data, output_data = run_batch_iterator(self.batch, input_data, self.processing_function, self.pre_processing_function)

                self.update_results(input_data, output_data)

                sampler.tell(output_data, input_data)
