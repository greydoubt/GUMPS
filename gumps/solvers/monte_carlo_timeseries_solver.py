# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Optional

import attrs
import numpy as np
import scipy.stats
import xarray as xr
import pandas as pd
from typing_extensions import Protocol

from gumps.solvers.batch_solver import AbstractBatchSolver

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class ProbabilityFunction(Protocol):
    "interface that requires a class has a ppf function"

    def ppf(self, percent_points:np.ndarray) -> np.ndarray:
        "percent point function"

@attrs.define
class MonteCarloParameters:
    "Interface for PyMoo solver required settings"
    variable_distributions: dict[str, ProbabilityFunction]
    target_probability: float
    window:int = 10
    min_steps: int = 4
    tolerance:float = 1e-2
    sampler_seed: Optional[int] = None
    sampler_scramble: bool = True
    runnable_batch_size: Optional[int] = None

class MonteCarloTimeSeriesSolver(AbstractBatchSolver):
    "pyMoo solver interface"

    def __init__(self, *, solver_settings: MonteCarloParameters):
        "initialize the monte carlo solver"
        self.solver_settings = solver_settings
        self.history: Optional[xr.Dataset] = None
        self.mean: Optional[xr.Dataset] = None
        self.max_difference: Optional[xr.Dataset] = None

        self.sampler = scipy.stats.qmc.Sobol(len(self.solver_settings.variable_distributions),
            seed=self.solver_settings.sampler_seed,
            scramble=self.solver_settings.sampler_scramble)

        self.delta, self.batch_size = self.calculate_batch_size()

    def calculate_batch_size(self) -> tuple[float, int]:
        "calculate the batch size"
        prob = np.array(self.solver_settings.target_probability)

        lower_delta = np.min(prob)
        upper_delta = np.min(1 - prob)
        delta = min(lower_delta, upper_delta)

        batch_size = int(1/delta)

        #based on poc/batch_size.ipynb if the batch size drops below about 64 convergence slows
        #this is an issue for high-probability regions, for low-probability regions the
        #batch size is already larger than this
        batch_size = max(batch_size, 64)

        #make sure the batch size is an even multiple of the runnable batch size if defined
        if self.solver_settings.runnable_batch_size is not None:
            batch_size = int(np.ceil(batch_size/self.solver_settings.runnable_batch_size)) * self.solver_settings.runnable_batch_size

        logger.info(f"Batch size: {batch_size}")

        return delta, batch_size


    def get_convergence(self) -> np.ndarray:
        "get convergence information"
        if self.history is None:
            raise RuntimeError("No history to calculate convergence")

        window = self.solver_settings.window

        if len(self.history['iteration']) > window:
            history = self.history.sel(iteration=slice(-window, None))
        else:
            history = self.history

        self.mean = history.mean(dim="iteration")
        self.max_difference = abs(history - self.mean).max(dim="iteration")

        mean_array = self.mean.to_array()
        max_difference_array = self.max_difference.to_array()
        upper_bound_array = mean_array + max_difference_array

        isclose = np.isclose(mean_array, upper_bound_array,
                             rtol=self.solver_settings.tolerance,
                             atol=self.solver_settings.tolerance)

        return isclose


    def has_next(self) -> bool:
        "check if there is another step"
        if self.history is None or len(self.history['iteration']) <= self.solver_settings.min_steps:
            return True

        isclose = self.get_convergence()

        return not isclose.all()

    def ask(self) -> pd.DataFrame:
        "return the population that needs to be evaluated"
        lower_probability = self.delta/100.0
        upper_probability = 1 - lower_probability

        sample = self.sampler.random(self.batch_size)

        sample_probability = sample * (upper_probability - lower_probability) + lower_probability

        temp = {name:distribution.ppf(row) for row, (name, distribution) in zip(sample_probability.T, self.solver_settings.variable_distributions.items())}
        return pd.DataFrame(temp)


    def tell(self, loss: xr.Dataset):
        "tell the system the results of calculations on the ask call"
        if self.history is None:
            self.history = xr.concat([loss, ], dim="iteration")
        else:
            self.history = xr.concat([self.history, loss], dim="iteration")
