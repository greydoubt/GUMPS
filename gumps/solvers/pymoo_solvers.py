# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import attrs
import warnings
import scipy.stats

from gumps.solvers.batch_solver import AbstractBatchSolver

#multiple objective
#from pymoo.algorithms.moo.age import AGEMOEA
#from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.unsga3 import UNSGA3
#single objective
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class AutoProblem(Problem):
    """create a basic optimization problem for the pymoo ask-tell
    interface with support for automatic transform bound"""

    def __init__(self, number_var, number_obj, lower_bound, upper_bound, auto_transform):
        if auto_transform:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            super().__init__(n_var=number_var, n_obj=number_obj, n_constr=0, xl=[0.0] * len(lower_bound), xu=[1.0] * len(upper_bound))
        else:
            super().__init__(n_var=number_var, n_obj=number_obj, n_constr=0, xl=lower_bound, xu=upper_bound)

    def _evaluate(self, x, out, *args, **kwargs):
        "this is here for pylint, the method is no longer used as part of the ask-tell interface"

@attrs.define
class PyMooSolverParameters:
    "Interface for PyMoo solver required settings"
    population_size: int
    algorithm_name: str
    number_var: int
    number_obj: int
    auto_transform: bool
    total_generations: int
    lower_bound: np.ndarray =  attrs.field(converter=np.array)
    upper_bound: np.ndarray =  attrs.field(converter=np.array)

class PyMooSolver(AbstractBatchSolver):
    "pyMoo solver interface"

    def __init__(self, *, solver_settings: PyMooSolverParameters):
        "initialize the pyMoo solver"
        self.solver_settings = solver_settings

        self.lower_bound = self.solver_settings.lower_bound
        self.upper_bound = self.solver_settings.upper_bound
        self.population_size = self.solver_settings.population_size
        self.algorithm_name = self.solver_settings.algorithm_name

        self.auto_problem = AutoProblem(number_var=self.solver_settings.number_var,
                number_obj=self.solver_settings.number_obj,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                auto_transform=self.solver_settings.auto_transform)

        self.algorithm = self.get_algorithm()
        self.algorithm.setup(self.auto_problem, termination=('n_gen',
            self.solver_settings.total_generations),
            verbose=False)
        self.current_population = np.empty([0])

    @property
    def starting_population(self):
        "get the starting population"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = scipy.stats.qmc.Sobol(len(self.lower_bound), scramble=False).random(n=self.population_size)

        return  sobol * (self.upper_bound - self.lower_bound) + self.lower_bound

    def get_evolution_strategy(self, val):
        """this is a callback which gets the evolution strategy for cma-es
        This callback can also be used to terminate the algorithm if it returns True"""
        self.es = val
        return False

    def get_algorithm(self):
        "get the selected agorithm"
        algorithm = self.algorithm_name
        init_pop = self.starting_population
        pop_size = init_pop.shape[0]
        n_obj = self.solver_settings.number_obj
        if algorithm == 'nsga2':
            return NSGA2(pop_size=pop_size,
                sampling=Population.new("X", self.starting_population))
        elif algorithm =='rnsga2':
            ref_points = get_reference_directions("energy", n_obj, min(pop_size, 1000))
            return RNSGA2(ref_points=ref_points,
                pop_size=pop_size,
                sampling=init_pop)
        elif algorithm == 'nsga3':
            ref_dirs = get_reference_directions("energy", n_obj, min(pop_size, 1000))
            return NSGA3(pop_size=pop_size,
                  ref_dirs=ref_dirs,
                  sampling=init_pop)
        elif algorithm == 'unsga3':
            ref_dirs = get_reference_directions("energy", n_obj, min(pop_size, 1000))
            return UNSGA3(ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=init_pop)
        elif algorithm == 'rvea':
            if n_obj == 1:
                raise ValueError(f"Algorithm {algorithm} not supported for single objective")
            else:
                ref_dirs = get_reference_directions("energy", n_obj, min(pop_size, 1000))
                return RVEA(pop_size=pop_size,
                    ref_dirs=ref_dirs,
                    sampling=init_pop)
        elif algorithm == 'smsemoa':
            return SMSEMOA(pop_size=pop_size,
                  sampling=init_pop)
        elif algorithm == 'cmaes':
            if n_obj > 1:
                raise ValueError(f"Algorithm {algorithm} not supported with multiple-objectives")
            else:
                return CMAES(popsize=pop_size,
                    x0=init_pop,
                    termination_callback=self.get_evolution_strategy)
        else:
            raise ValueError(f"Algorithm {algorithm} is not supported")

    def has_next(self) -> bool:
        return self.algorithm.has_next()

    def ask(self) -> np.ndarray:
        self.current_population = self.algorithm.ask()
        return np.array([i.x for i in self.current_population])

    def tell(self, loss: np.ndarray):
        for ind, error in zip(self.current_population, loss):
            ind.F = error

        self.algorithm.tell(infills=self.current_population)

