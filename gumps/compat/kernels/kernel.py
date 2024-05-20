# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.common import Node
from gumps.compat.variables import VariableRegistry
import logging

logger = logging.getLogger(__name__)
class Kernel(Node):
    '''
    Adapter between Pythonic Models and Engine

    Parameters
    -----------
    model: Model
        Model to be run
    Engine: Engine
        Engine to be run on
    name: String
        Default -> None
    '''
    def __init__(self, model, engine, name=None):
        super().__init__(name)
        self._model = model
        self._engine = engine
        self._addChildren([model,engine])
        self.parent_study = None

    def initialize(self):
        """Initializes Kernel"""
        pass

    def deinitialize(self):
        """Deinitializes Kernel"""
        pass

    def run(self):
        """
        The basic implementation: simply call the engine with the kernel.
        """
        logger.debug('running {}'.format(self._model.id))
        self._engine.run(self)
        logger.debug('finished running {}'.format(self._model.id))
        self._results = self.get_variables()
        logger.debug('finished getting results {}'.format(self._model.id))
        # self._response = self._engine.get_variables(self)


    @property
    def model(self): return self._model

    @property
    def study(self): return self.parent_study

    def get_variables(self):
        """Returns variables for itself and all subclasses
        
        Parameters
        -----------
        None
        Returns
        -------
        Variable Registry
        """

        variables = VariableRegistry(self)
        #Register itself
        variables.register(self._variables)
        for _ , child in self._children.items():
            variables.register(child.get_variables())
        return variables

    def get_response(self):
        return self._results.outputs()