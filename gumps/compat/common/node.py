# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from .container import Container
from itertools import groupby

class Node(Container):
    """
    Element in GUMPS tree with an indeterminant number of children.

    Parameters
    -----------
    name: String - 
        Default -> None
    variable: List of Variables 
        Default -> []

    Attributes
    -----------
    id : String
        Name of Element, critical as this name is how we can traverse between elements to set variables.
    _variables : VariableRegistry
        Container that aggregates all variables.
    _children : Dictionary with key value pairs (id, Container)
        Contains references to each element's children.
    """
    def _get_variables_by_container(self, variables):
        reformed = {}
        seen = []
        for key, group in groupby(variables, lambda x: x.symbol[0]):
            if not key in seen:
                reformed[key] = []
                seen.append(key)
            for variable in group:
                reformed[key].append(variable)
        return reformed
    
    def set_variables(self, variables):
        '''Sets variables for container.
        First groups together variables by their parent id to reduce number of times 
        we traverse the tree. Afterwards it loops through each nodeID to get a reference
        of the container/node and then calls set variables with its own variables.

        Parameters
        ----------
        variables : list of variables to be set.

        Returns
        -------
        None
        '''
        groupedVariables = self._get_variables_by_container(variables)
        for nodeId, variables in groupedVariables.items():
            node = self._findId(nodeId)
            if node:
                node.set_variables(variables)
            else: #assumes that the nodeID is itself, clean up
                self._variables.assign_value(variables)