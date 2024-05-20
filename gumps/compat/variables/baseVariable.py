# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.variables import VariableTypeRegister

class Variable(VariableTypeRegister):
    """
    The structure of all variables used in the model.
    """
    use = None

    def __init__(self, symbol,
                       name = None,
                       description = None,
                       factor = False,
                       response = False,
                       units = None,
                       value = None):
        self.symbol = symbol
        self.name = name
        self.description = description
        self.factor = factor
        self.response = response
        self.units = units
        self.value = value

    def __repr__(self):
        return f'{self.symbol}: {self.value}'

    def __iter__(self):
        yield self

    def copy(self):
        return Variable(self.symbol, self.name, self.description,
                        self.factor, self.response, self.units, self.value)

    def change_symbol(self, newSymbol):
        self.symbol = newSymbol