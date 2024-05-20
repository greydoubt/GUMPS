# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import abc as interface

class AbstractSolver(interface.ABC):
    """
    An abstract solver class. A solver defines and executes the algorithm used to solve a problem
    """

    @interface.abstractclassmethod
    def presolve(self, structure):
        pass

    @interface.abstractclassmethod
    def solve(self, structure):
        pass

    @interface.abstractclassmethod
    def postsolve(self, structure):
        pass
