# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#This module uniformly samples the parameter space using a Sobol or Latin Hypercube sampler

import logging

import attrs
import pandas as pd
import xarray as xr
import scipy

from gumps.solvers.batch_solver import AbstractBatchSolver

from gumps.scalers.log_combo_scaler import LogComboScaler
from gumps.scalers.minmax_scaler import MinMaxScaler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class SamplerSolverParameters:
    "Interface for Sampler required settings"
    number_of_samples: int
    lower_bound: pd.Series = attrs.field(converter=pd.Series)
    upper_bound: pd.Series = attrs.field(converter=pd.Series)

    sampler: str = attrs.field(default="sobol")
    @sampler.validator
    def check(self, attribute, value):
        if value not in {"sobol", "latin"}:
            raise ValueError(f"{attribute.name} = {value} is not allowed, expected sobol, latin")

class SamplerSolver(AbstractBatchSolver):

    "Sampler solver interface"
    def __init__(self, *, solver_settings: SamplerSolverParameters):
        "initialize the SamplerSolverParameters"
        self.solver_settings = solver_settings
        self.completed: bool = False
        self.scaler = LogComboScaler(MinMaxScaler())

        self.scaler.fit(pd.DataFrame([self.solver_settings.lower_bound, self.solver_settings.upper_bound]))

    def has_next(self) -> bool:
        return not self.completed

    def ask(self) -> pd.DataFrame:
        "Return the array for sampling"
        if self.solver_settings.sampler == "sobol":
            sample = scipy.stats.qmc.Sobol(len(self.solver_settings.lower_bound), scramble=False).random(n=self.solver_settings.number_of_samples)

        elif self.solver_settings.sampler == "latin":
            sample = scipy.stats.qmc.LatinHypercube(len(self.solver_settings.lower_bound), scramble=False).random(n=self.solver_settings.number_of_samples)

        data = self.scaler.inverse_transform(pd.DataFrame(sample, columns=self.solver_settings.lower_bound.index))
        return data

    def tell(self, loss: pd.DataFrame|xr.Dataset):
        if loss is not None:
            self.completed = True
