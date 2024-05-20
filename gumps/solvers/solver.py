# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Abstract solver interface for running a simulation."

import copy
import logging
from abc import ABCMeta, abstractmethod
from typing import Callable

import attrs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class SolverParameters:
    "Parameters for the solver"

class AbstractSolver(metaclass=ABCMeta):
    "this is an abstract solver class"
    def __init__(self, *, problem: dict, solver_settings: SolverParameters|None = None):
        """initialize the solver with the problem and solver settings"""
        self._problem = problem

        if solver_settings is None:
            solver_settings = SolverParameters()
        self.solver_settings = solver_settings

    @classmethod
    def create_solver(cls, *, problem: dict, solver_settings: SolverParameters|None = None):
        """create a new solver for the current class
        This simplifies new_solver method for creating subclasses correctly"""
        return cls(problem = problem, solver_settings=solver_settings)

    def new_solver(self, data:dict):
        """create a new solver using this data
        Python 3.11: The return type for this function should be SelfType"""
        problem = copy.deepcopy(self._problem)
        problem.update(data)
        return self.create_solver(problem = problem, solver_settings=self.solver_settings)

    @property
    def problem(self):
        "make a copy of the problem before returning it"
        logger.debug("making a copy of self._problem")
        return copy.deepcopy(self._problem)

    @abstractmethod
    def solve(self, f:Callable, save_function: Callable):
        pass
