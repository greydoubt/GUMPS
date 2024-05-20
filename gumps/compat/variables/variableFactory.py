# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.variables import *

class VariableFactory():
    """
    Method to create variables

    If no use is passed it will create a default variable
    In order to create a certain variable type, pass in the
    name of the variable in all lowercase as the use

    Parameter:
    use: string
        Type of Variable to return.
    Returns:
        variable
    """
    @staticmethod
    def create(symbol, use='None', **kwargs):
        """
        Simple Factory to create variables of different uses.
        """
        kwargs['symbol'] = symbol
        use = use.lower()
        if use == 'time':
            return TimeVariable(**kwargs)
        elif use == 'state':
            return StateVariable(**kwargs)
        elif use == 'input':
            return InputVariable(**kwargs)
        elif use == 'output':
            return OutputVariable(**kwargs)
        elif use == 'param':
            return ParamVariable(**kwargs)
        elif use == 'none':
            return  Variable(**kwargs)
        else:
            raise AttributeError('{} type does not exist'.format(use))

    @staticmethod
    def batch(varDict):
        """
        Function to create multiple copies of each type of variable loaded in through a
        python dictionary of the form
        {
            "varType" : [{
                        "varAttribute1" : varAttribute1_value
            }]
        }

        Example ::
        {
            "input" : [{
                "value" : 0,
                "symbol" : "CHO"
            },
            {
                "value" : 1,
                "symbol" : "H20"
            }
            ]
        }
        """
        varList = []
        for varType, varsToAdd in varDict.items():
            varList.extend([VariableFactory.create(use=varType, **varAttr) for varAttr in varsToAdd])
        return varList