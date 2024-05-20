# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
import numpy as np
import pandas as pd
import scipy.stats
import attrs
from typing_extensions import Protocol
from gumps.solvers.batch_solver import AbstractBatchSolver

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class ProbabilityFunction(Protocol):
    "interface that requires a class has a ppf function"

    def ppf(self, percent_points:np.ndarray) -> np.ndarray:
        "percent point function"
        ...

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

class MonteCarloSolver(AbstractBatchSolver):
    "pyMoo solver interface"

    def __init__(self, *, solver_settings: MonteCarloParameters):
        "initialize the monte carlo solver"
        self.solver_settings = solver_settings
        self.history: Optional[pd.DataFrame] = None
        self.mean: Optional[pd.Series] = None
        self.max_difference: Optional[pd.Series] = None

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

        #based on poc/batch_size.ipynb if the batch size drops below about 1024 convergence slows
        #this is an issue for high-probability regions, for low-probability regions the
        #batch size is already larger than this
        batch_size = max(batch_size, 1024)

        #make sure the batch size is an even multiple of the runnable batch size if defined
        if self.solver_settings.runnable_batch_size is not None:
            batch_size = int(np.ceil(batch_size/self.solver_settings.runnable_batch_size)) * self.solver_settings.runnable_batch_size

        logger.info(f"Batch size: {batch_size}")

        return delta, batch_size


    def has_next(self) -> bool:
        "check if there is another step"
        if self.history is None or self.history.shape[0] <= self.solver_settings.min_steps:
            return True
        window = self.solver_settings.window

        if len(self.history) > window:
            history = self.history.iloc[-window:,:]
        else:
            history = self.history

        self.mean = history.mean()
        self.max_difference = (history - self.mean).abs().max()

        isclose = np.isclose(self.mean, self.mean + self.max_difference,
                             rtol=self.solver_settings.tolerance,
                             atol=self.solver_settings.tolerance)

        return not isclose.all()

    def ask(self) -> pd.DataFrame:
        "return the population that needs to be evaluated"
        lower_probability = self.delta/100.0
        upper_probability = 1 - lower_probability

        sample = self.sampler.random(self.batch_size)

        sample_probability = sample * (upper_probability - lower_probability) + lower_probability

        temp = {name:distribution.ppf(row) for row, (name, distribution) in zip(sample_probability.T, self.solver_settings.variable_distributions.items())}
        return pd.DataFrame(temp)


    def tell(self, loss: pd.Series):
        "tell the system the results of calculations on the ask call"
        if self.history is None:
            self.history = loss.to_frame().T
        else:
            self.history = pd.concat([self.history, loss.to_frame().T],ignore_index=True)