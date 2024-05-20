# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.variables import Variable

class InputVariable(Variable):
    """General Purpose Input, vectorizable"""
    use = 3
    def __init__(self, symbol,
                       name = None,
                       description = None,
                       factor = False,
                       response = False,
                       units = None,
                       value = None):
        super().__init__(symbol, name, description,
                        factor, response, units, value)
                
    def copy(self):
        return InputVariable(self.symbol, self.name, self.description,
                        self.factor, self.response, self.units, self.value)
