# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.abstractclasses import AbstractSolver
from gumps.compat.variables import VariableRegistry
from gumps.compat.common import Container
import numpy as np

class Solver(AbstractSolver, Container):
    """
    A base solver class. A solver defines and executes the algorithm used to
    solve a problem
    """
    #TODO: currently, ODE integration is handled by an engine (e.g.
    #ScipyODEIntEngine) because the method of integration is independent of
    #the problem and solver. Consider moving integration to the solver.

    def store(self, variables):
        """A method that stores variables as a new row in a pandas DataFrame. 
        The problem will specify which variables are to be stored.
        :variables: a VariableRegistry containing the variables to store."""
        for k, v in variables.items():
            try:
                s = self._variables.get(k)
                s.value = np.vstack((s.value, v.value))
            except KeyError:
                self._variables.set(k, v.copy())
            except ValueError:
                print("ValueError storing {} in {} with value {}".format(v.value, (self.id, k), s.value))
                raise

    def store_reset(self):
        self._variables = VariableRegistry(self, self._initial_variables)

    def presolve(self, problem, structure):
        self.store_reset()
        
    def solve(self, problem, structure):
        structure.run()

    def postsolve(self, problem, structure):
        pass

