# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import pandas as pd
from typing import Optional, Callable
from pathlib import Path

from gumps.common import hdf5
from gumps.common.app_utils import run_batch_iterator
from gumps.studies.batch_study import AbstractBatchStudy
import gumps.solvers.sampler

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class ParametricSweepApp:
    "Create a parametric sweep app"

    def __init__(self, *, parameters: gumps.solvers.sampler.SamplerSolverParameters,
        processing_function: Callable, directory:Optional[Path]=None,
        batch: AbstractBatchStudy,pre_processing_function: Callable|None=None):
        "initialize the Parametric Sweep app"
        self.parameters = parameters
        self.processing_function = processing_function
        self.pre_processing_function = pre_processing_function
        self.batch = batch

        self.directory = directory
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

        self.factors: pd.DataFrame | None = None
        self.responses: pd.DataFrame | None = None

    def answer(self) -> pd.DataFrame:
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
        data.root.responses = self.responses.to_numpy()
        data.root.factor_names = self.factors.columns
        data.root.response_names = self.responses.columns
        data.save()

    def run(self):
        solver = gumps.solvers.sampler.SamplerSolver(solver_settings=self.parameters)
        with self.batch:
            while solver.has_next():
                factors = solver.ask()

                factors, responses = run_batch_iterator(self.batch, factors, self.processing_function, self.pre_processing_function)

                solver.tell(responses)

                self.factors = factors
                self.responses = responses
