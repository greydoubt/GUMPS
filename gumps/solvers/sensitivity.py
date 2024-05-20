# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"create a solver for sensitivity analysis that uses SALib"

import logging

import attrs
import matplotlib.axes
import numpy as np
import pandas as pd
import SALib.util.results
import SALib.analyze.sobol
import SALib.sample.sobol

from gumps.solvers.batch_solver import AbstractBatchSolver

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class SensitivitySolverParameters:
    "parameters for the sensitivity solver"
    lower_bound: pd.Series
    upper_bound: pd.Series
    sample_power: int = 5
    calc_second_order: bool = True
    seed: int|None = None

    @property
    def total_points(self) -> int:
        "return the total number of points that will be sampled"
        if self.calc_second_order:
            return self.N * (2*self.D + 2)
        else:
            return self.N * (self.D + 2)

    @property
    def N(self) -> int:
        "this is the N number for SALib and designed to make the interface easier to understand"
        return 2**self.sample_power

    @property
    def D(self) -> int:
        "this is the D number for SALib and designed to make the interface easier to understand"
        return len(self.lower_bound)

    def get_problem(self) -> dict:
        "return a problem in the form that SALib needs it"
        problem = {
            'num_vars': self.D,
            'names': self.lower_bound.index.tolist(),
            'bounds': [[lb,ub] for lb,ub in zip(self.lower_bound, self.upper_bound)]
        }
        return problem



class SensitivitySolver(AbstractBatchSolver):
    "Sensitivity solver interface"

    def __init__(self, *, solver_settings: SensitivitySolverParameters):
        "initialize the sensitivity solver"
        self.solver_settings = solver_settings
        self._has_next: bool = True
        self.analysis: SALib.util.results.ResultDict|None = None
        self.sample_length: int|None = None


    def has_next(self) -> bool:
        return self._has_next


    def ask(self) -> np.ndarray:
        samples = SALib.sample.sobol.sample(problem=self.solver_settings.get_problem(),
                N = self.solver_settings.N,
                calc_second_order=self.solver_settings.calc_second_order,
                seed=self.solver_settings.seed)
        self.sample_length = len(samples)
        return samples


    def tell(self, loss: np.ndarray) -> None:
        "tell the solver the loss"
        if self.sample_length is None:
            raise RuntimeError("You must call ask() before you can call tell()")

        if len(loss) != self.sample_length:
            raise ValueError("The length of the loss does not match the length of the samples")

        self.analysis = SALib.analyze.sobol.analyze(
            problem=self.solver_settings.get_problem(),
            Y=loss,
            calc_second_order=self.solver_settings.calc_second_order)

        self._has_next = False

    def plot(self) -> np.ndarray[matplotlib.axes._axes.Axes]:
        "plot the SALib results"
        if self.analysis is None:
            raise RuntimeError("You must call tell() before you can plot()")
        else:
            return self.analysis.plot()
