# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.solvers import AbstractSolver
import gumps.solvers.simple_solver
from gumps.kernels import AbstractKernel

from gumps.common import IllDefinedException

import pandas as pd
import numpy as np
import attrs
from abc import ABCMeta, abstractmethod

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class AbstractStudy(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        "run the study"

    @abstractmethod
    def state_frame(self) -> pd.DataFrame:
        "return the states as a pandas dataframe"

    @abstractmethod
    def run_data(self, data:dict) -> pd.DataFrame:
        "run a new simulation with data merged into the kernel and output states converted to a dataframe"

class SimulationStudy(AbstractStudy):
    "create a simple simulation study"
    def __init__(self, solver:AbstractSolver, kernel: AbstractKernel):
        self.solver = solver
        self.kernel = kernel

        self.states = []
        self.result = None
        self.check_problem()

    def save(self, state):
        "save the current state"
        logger.debug("saving the state  %s", state)
        self.states.append(state)

    def check_problem(self):
        "check if the problem is valid based on the variables"
        allowed = self.kernel.allowed_state
        problem = set(self.solver.problem)
        logger.debug("study allowed variables: %s  problem variables: %s", allowed, problem)
        if not problem.issubset(allowed):
            raise IllDefinedException("Missing ", problem - allowed)

    def run(self):
        "run the study"
        logger.debug("running the study")
        self.result = self.solver.solve(self.kernel.f, self.save)
        logger.debug("study has completed with result: %s", self.result)
        return self.result

    def state_frame(self) -> pd.DataFrame:
        "return the states as a pandas dataframe"
        return pd.DataFrame([attrs.asdict(state) for state in self.states])

    def run_data(self, data:dict) -> pd.DataFrame:
        "run a new simulation with data merged into the kernel and output states converted to a dataframe"

        #Not sure what the best way to handle this is
        #the new variables could require some expensive functions to be cached
        #such as setting up cvxopt and the initial matrixes
        #but they are also the problem variables
        new_kernel = self.kernel.new_kernel(data)
        new_solver = self.solver.new_solver(data)
        study = SimulationStudy(new_solver, new_kernel)
        study.run()
        return study.state_frame()

class SimpleSimulationStudy(SimulationStudy):
    def __init__(self, problem:dict, kernel:AbstractKernel):
        solver = gumps.solvers.simple_solver.SimpleSolver(problem=problem, solver_settings=None)
        super().__init__(solver, kernel)


@attrs.define
class TestState:
    "states for TestStudy"
    x:int
    y:int

class TestStudy(AbstractStudy):
    """This is a simple test study that takes x as an argument
    and return x if x is positive and an exception if negative"""

    def __init__(self, x:int):
        self.x = x

    def run(self):
        if self.x < 0:
            raise RuntimeError("x is negative")
        return TestState(x=self.x, y=self.x)

    def state_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{'x': self.x, 'y': self.x}])

    def run_data(self, data:dict) -> pd.DataFrame:
        if data['x'] < 0:
            raise RuntimeError("x is negative")
        elif data['x'] > 10:
            return pd.DataFrame([{'x': data['x'], 'y': np.nan}])
        else:
            return pd.DataFrame([{'x': data['x'], 'y': data['x']}])

class TestStudyTime(AbstractStudy):
    """This is a simple test study for creating time series data and
    corresponds with the batch_time_study_example"""

    def __init__(self, model_variables:dict, problem:dict):
        self.model_variables = model_variables
        self.time_series = np.linspace(self.model_variables['time_start'],
                                       self.model_variables['time_end'],
                                       self.model_variables['time_points'])
        self.problem = problem
        self.frame : pd.DataFrame| None = None

    def run(self) -> pd.DataFrame:
        frame = self.run_data(self.problem)
        self.frame = frame
        return frame

    def state_frame(self) -> pd.DataFrame:
        if self.frame is None:
            raise RuntimeError("No data has been run")
        return self.frame

    def run_data(self, data:dict) -> pd.DataFrame:
        temp = {}
        temp['time'] = self.time_series

        nan = True if data.get('nan', 0.0) > 0.9 else False
        fail = True if data.get('fail', 0.0) > 0.9 else False

        if 'fail' in data:
            del data['fail']

        if 'nan' in data:
            del data['nan']

        for key,value in data.items():
            if nan:
                temp[f"t{key}"] = np.nan * np.ones(self.time_series.shape)
            elif fail:
                raise RuntimeError("fail")
            else:
                temp[f"t{key}"] = value * self.time_series + value
        return pd.DataFrame(temp).set_index('time')