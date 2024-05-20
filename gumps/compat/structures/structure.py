# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.abstractclasses import AbstractStructure
from gumps.compat.common import Node
from gumps.compat.variables import VariableRegistry
from .structure_graph import StructureGraph
import logging

logger = logging.getLogger(__name__)
class Structure(Node, AbstractStructure):
    """
    The base Structure, allows substructures and connections to be appended.

    Exists to impose the order in which the containers need to be executed.

    Attributes:
    _study: Study
        Parent Study of the Structure
    _graph: StructureGraph
        object that holds graph of nodes.
    """

    def __init__(self, study, name=None):
        self._study = study
        self._graph = StructureGraph()
        super().__init__(name)

    def add_element(self, element):
        """Add element to structure graph"""
        element.parent_study = self._study
        self._graph.add_node(element)
        self._addChildren([element])

    def add_connection(self, connection):
        """Add connection to structure graph"""
        self._graph.add_edge(connection)

    def _get_targets(self):
        """Gets targets from every connection"""
        targets = []
        connections = self._graph.get_connections()
        for connection in connections:
            targets.extend(connection.get_all_vp_targets())
        return targets

    def order(self):
        """Returns graph's order"""
        return self._graph.order()

    def initialize(self):
        """Initializes every element on the graph in order"""
        for element in self._graph.order():
            element.initialize()

    def deinitialize(self):
        """Deinitializes every element on the graph in order"""
        for element in self._graph.order():
            element.deinitialize()

    def run(self):
        """Runs every element on the graph in order

        After every run, propogate any connections in place"""
        for element in self._graph.order():
            element.run()
            self._graph.propagate(element)

    def get_variables(self):
        """Get variables
        Returns:
        Variable registry with all variables in the structure
        """
        variables = VariableRegistry(self._study)
        for element in self._graph.order():
            variables.register(element.get_variables())
        return variables

    def get_response(self):
        """Get response from the last element in the graph"""
        self._response = VariableRegistry(self._study)
        logger.debug('{} has elements {}'.format(self._study.id, [element.id for element in self._graph._graph]))
        for element in self._graph.terminus():
            logger.debug('getting response from {} ({}) with attributes {}'.format(element.id,element,element.__dict__.items()))
            self._response.register(element.get_response())
            logger.debug('total response is now {}'.format(self._response))
        return self._response
