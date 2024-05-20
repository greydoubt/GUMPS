# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import OrderedDict
from gumps.compat.variables import VariableTypeRegister

class VariableRegistry:
    """
    A container for model variables, providing reference utilities. For all
    queries, returns another VariableRegistry with a subset of variables.
    :parent: The container in which the variable registry resides. Must be a 
    subclass of Container.
    :variables: A Variable, a VariableRegistry or a list of Variables to
    initialize this variable registry with
    """
    def __init__(self, parent, variables = None):
        self._parent = parent
        self._variables = OrderedDict()
        if variables is None:
            variables = []
        self.add(variables)
        self.varType = VariableTypeRegister()._registry

    def __getitem__(self, symbol):
        #A succinct reference to the value of a variable from the variable
        #registry, for use in the model equations
        return self.get(symbol).value

    def __setitem__(self, symbol, value):
        self.get(symbol).value = value

    def __iter__(self):
        for key in self._variables:
            yield self._variables[key]

    def __len__(self):
        return len(self._variables)

    def __repr__(self):
        return repr(self._variables)

    def _makekey(self, key):
        return (self._parent.id, key)

    def _checkfor(self, key):
        if key in self._variables:
            msg = f'Attempting to add variable {key} to a ' \
                  f'VariableRegistry which already contains it.'
            raise ValueError(msg)

    def items(self):
        return self._variables.items()

    def values(self):
        return [self[key] for key in self._variables]

    def keys(self):
        return [key for key in self._variables]

    def symbols(self):
        return [v.symbol for k, v in self._variables.items()]

    def get(self, symbol):
        #Since we've overridden __getitem__ to return the value of the
        #variable, this is required to return the variable itself.
        try:
            key = self._makekey(symbol)
            return self._variables[key]
        except KeyError:
            return self._variables[symbol]

    #TODO: Lots of code in common with get(), but I don't see how I can reuse
    #it when I need to be setting a reference here
    def set(self, symbol, variable):
        #Since we've overridden __setitem__ to set the value of the
        #variable, this is required to set the variable itself.
        try:
            key = self._makekey(symbol)
            self._variables[key] = variable
        except KeyError:
            self._variables[symbol] = variable


    def add(self, other):
        """
        Add a variable, a list of variables, or the variables in a
        VariableRegistry to this registry. Their symbols will be prepended to
        include this variable registry's parent. Raises an exception if the
        variable already exists.
        :other: A Variable or a list of variables. 
        """
        for v in other:
            key = self._makekey(v.symbol)
            self._checkfor(key)
            self._variables[key] = v

    def register(self, other):
        """
        Register the variables in a VariableRegistry with this variable
        registry, keeping the parent ID from the source registry.
        Raises an exception if one of the variables already exists.
        :other: A VariableRegistry
        """
        for key, var in other.items():
            self._checkfor(key)
            self._variables[key] = var#.update({key: var})

    def assign_value(self, variable):
        """
        Assigns the passed variable's value (or VariableRegistry's members'
        values) to the corresponding variable(s) in this VariableRegistry
        :variable: Variable, list of Variables or VariableRegistry.
        """
        for v in variable:
            myvariable = self.get(v.symbol)
            myvariable.value = v.value

    def subset(self, keys):
        vr = VariableRegistry(self._parent)
        if isinstance(keys, (str, tuple)):
            keys = [keys]
        for key in keys:
            vr.set(key, self.get(key))
        return vr

    def _filter(self, filter_on, filter_val):
        filt = lambda x: getattr(x, filter_on) == filter_val
        vr = VariableRegistry(self._parent)
        vr.register({k: v for k, v in self.items() if filt(v)})
        return vr

    def states(self):
        return self._filter('use', self.varType['StateVariable'])

    def time(self):
        return self._filter('use', self.varType['TimeVariable'])

    def inputs(self):
        return self._filter('use', self.varType['InputVariable'])

    def outputs(self):
        return self._filter('use', self.varType['OutputVariable'])

    def params(self):
        return self._filter('use', self.varType['ParamVariable'])

    def factors(self):
        return self._filter('factor', True)

    def responses(self):
        return self._filter('response', True)

    def uses(self):
        uses = {'output': [], 'input':[] }
        for var in self:
            if var.use == self.varType['OutputVariable']:
                uses['output'].append(var.symbol)
            else:
                uses['input'].append(var.symbol)
        return uses

    def convertToDict(self):
        """
        Converts Variable Registry into a normal dictionary,
        useful when unpacking into external function calls
        """
        newDict = {}
        for var in self:
            newDict[var.symbol] = var.value
        return newDict