# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.abstractclasses import AbstractEngine
from gumps.compat.common import Container

class Engine(Container, AbstractEngine):
    """
        Engine responsible for running pythonic Model Equation

        Parameters
        -----------
        name: String
            Default -> None
    """
    def set_var_dict(self, kernel, source_variables):
        kernel.model.set_var_dict(source_variables)

    def initialize(self):
        pass

    def run(self, kernel):
        '''Runs model's equation.

        Parameters
        ----------
        source_variables : list of variables to be set.

        Returns
        -------
        None

        TODO
        ------
        Refactor to not pass in the kernel, but a function for it to run
        '''
        kernel.model.equation()
