# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Create a simple iterative solver"

import logging
from typing import Callable
import copy

import numpy as np
import attrs
from gumps.solvers.solver import AbstractSolver, SolverParameters

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class IterativeSolverParameters(SolverParameters):
    "Parameters for the iterative solver"
    time_start: float
    time_end: float
    time_points: int

class IterativeSolver(AbstractSolver):
    "This is a simple iterative solver"
    problem: dict
    solver_settings: IterativeSolverParameters

    def solve(self, f:Callable, save_function: Callable):
        for time in np.linspace(self.solver_settings.time_start, self.solver_settings.time_end, self.solver_settings.time_points):
            problem = copy.copy(self.problem)
            problem['time'] = time
            state = f(problem)
            save_function(state)
        return state
