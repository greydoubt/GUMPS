# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import uuid
from gumps.compat.variables import VariableRegistry


class Container:
    """
    Element in GUMPS tree.
    Responsible for maintaining structure and 
    providing access methods to move throughout tree.

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


    def __init__(self, name = None, variables = []):
        self.id = uuid.uuid1() if name is None else name
        self._initial_variables = variables
        self._variables = VariableRegistry(self, variables)
        self._children = {}

    def __hash__(self):
        return hash((self.id))

    def __eq__(self, other):
        return self.id == other.id

    def _addChildren(self, children):
        for child in children:
            self._children[child.id] = child

    def get_variables(self):
        '''Gets variables for container

        Parameters
        ----------
        None

        Returns
        -------
        Variable Registry for this container
        '''
        return self._variables
    
    def _inspect_variables(self):
        return self._variables.uses()

    def set_variables(self, source_variables):
        '''Sets variables for container.

        Parameters
        ----------
        source_variables : list of variables to be set.

        Returns
        -------
        None
        '''
        self._variables.assign_value(source_variables)

    def _findUtil(self, node, nodeId):
        children = node._children
        for childId in children:
            if childId == nodeId:
                return children[childId]
            found = self._findUtil(children[childId], nodeId)
            if found:
                return found

    def _findId(self, nodeId):
        return self._findUtil(self, nodeId)

    def _checkLevel(self, queue, order):
        s = queue.pop(0)
        order.append(s.id)
        for _ , node in s._children.items():
            queue.append(node)
        return queue, order

    def _BFS(self):
        order = []
        queue = [self]
        while queue:
            levelSize = len(queue)
            while levelSize > 0:
                queue, order = self._checkLevel(queue, order)
                levelSize -= 1
            order.append('\n')
        return order

    def _print(self):
        order = self._BFS()
        for nodeId in order:
            print(nodeId, end=' ')
        print()
