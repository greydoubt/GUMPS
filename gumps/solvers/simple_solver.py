# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Create a simple solver"

import logging
from typing import Callable

from gumps.solvers.solver import AbstractSolver

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class SimpleSolver(AbstractSolver):
    "This is a simple solver"

    def solve(self, f:Callable, save_function: Callable):
        state = f(self.problem)
        save_function(state)
        return state
