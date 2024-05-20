# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.structures import Structure
from gumps.compat.common import Node
import logging


logger = logging.getLogger(__name__)
class Study(Node):
    """
    Core Node in the GUMPS tree

    Attributes:
    -----------
        _problem: Problem
            Responsible for holding state at initiation.
        _solver: Solver
            Responsible for defining how structure is run
        _structure: Structure
            Responsible for defining order of how elements are run
    """

    def __init__(self, problem, solver, structure = None, name=None):
        super().__init__(name)
        self._problem = problem
        self._solver = solver
        self._structure = Structure(self) if structure is None else structure
        self._addChildren([self._problem, self._solver, self._structure])
        self._state = None
        self.parent_study = None

    def initialize(self):
        """Injects state from problem and then initializes"""
        problemVars = self._problem._initial_variables
        self.set_variables(problemVars)
        self._structure.initialize()

    def deinitialize(self):
        """Deinitializes study"""
        self._structure.deinitialize()

    def run(self):
        """
        Solve the problem with the method defined by the solver. Populate the
        results.

        Return:
        None
        """
        self._solver.presolve(self._problem, self._structure)
        logger.debug('solving {}'.format(self.id))
        self._solver.solve(self._problem, self._structure)
        self._solver.postsolve(self._problem, self._structure)
        logger.debug('getting response from {}'.format(self.id))
        self._response = self._structure.get_response()
        logger.debug('{} response is {}'.format(self.id, self._response))

    def get_variables(self):
        """Gets all variables from structure"""
        return self._structure.get_variables()

    #FUTURE: we may want some distinction between all variables, solutions of
    #the solver, and responses of the models

    def get_children(self):
        """Gets all children from structure"""
        return self._structure._children
        
    def get_response(self):
        """Gets response"""
        logger.debug('getting response from study {} ({})'.format(self.id, self))
        return self._response

    @property
    def problem(self): return self._problem

    @property
    def solver(self): return self._solver

    @property
    def structure(self): return self._structure


