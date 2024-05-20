# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.common import Container
from gumps.compat.variables import VariableRegistry
from functools import wraps


class Problem(Container):
    """
    A base problem class which returns a True stop condition and does nothing
    when applied.
    TODO: extend problem application to set the uses of a model's variables;
    i.e. a problem defines whether a variables is a factor, response, state,
    or parameter. The model only defines which variables are inputs and which
    outputs to its equation.
    """
    def get_factors(self):
        pass

    def apply(self, problemFormulation, structureVars):
        problemVars = VariableRegistry(self)
        for modelVarSym, solverVarSym in problemFormulation.items():
            problemVar = structureVars.get(modelVarSym).copy()
            problemVar.change_symbol(solverVarSym)
            problemVars.add(problemVar)
        return problemVars