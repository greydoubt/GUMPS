# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import networkx as nx


class StructureGraph:
    """
    Contains the definition of a structure's directed acyclic graph and implements methods for querying the graph's properties.

    Attributes

    -----------

    _graph :  dict
        Directed graph in the forward version
    _rgraph : dict
        Directed graph in the reverse direction
    """

    def __init__(self):

        self._adj = None
        self._graph = nx.DiGraph()
        self._order_cache = None

    def order(self):
        """
        Returns the order in which nodes may be visited to prevent backtracking
        Return: 
        A dictionary with container key,value pairs
        """
        if self._order_cache is None:
            self._order_cache = list((nx.topological_sort(self._graph)))
        return self._order_cache

    def terminus(self):
        """
        return a list of the last node(s) in the path. If the path diverges, the list will have more than one member.
        """
        return [x for x in self._graph.nodes() if self._graph.out_degree(x) == 0]

    def add_node(self, study):
        """
        adds the study as a node in the structure graph
        Parameters:
        -----------
          study: The study to be added as a node
        Returns:
        None
        """
        self._graph.add_node(study)

    def add_edge(self, connection):
        """
        adds a connection between two nodes in the structure graph
        Parameters:
        -----------
            connection: A Connection object with target and source attributes referring to the target and source studies
        Returns:
        None
        """
        target = connection.target
        source = connection.source
        self._graph.add_edge(source, target, connection=connection)

    def get_connections(self):
        """Gets all connections
        Returns:
        List of Connections
        """
        return [e for e in self._graph.edges]

    def propagate(self, source):
        """
        propagate data from a study to the targets of its connections
        
        Parameters:
        ------------
        source:
            the source study from which to propagate data
        """

        for edge in self._graph.edges(source, data=True):
            for pair in edge[2]['connection'].variable_pairs:
                pair.apply_forward()
