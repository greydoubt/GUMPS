# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import abc as interface



class AbstractStructure(interface.ABC):
    """
    A structure is a collection (graph) of studies, defining the data connections between their kernels and the sequence of solving their associated problems. A structure may be nested, so its children may be structures or studies, and the graph is acyclic but may include unidirectional or bidirectional connections (associated with problems which may be solved serially or simultaneously, the practical behavior of the latter being that the solutions of bidirectionally coupled problems must converge at each time step).
    """

    @interface.abstractclassmethod
    def add_element(self, study):
        """
        Used to add a substructure or study to this structure.
        Structures may be nested to any depth.
        """
        pass
        # self._graph.add_node(study)

    @interface.abstractclassmethod
    def add_connection(self, connection):
        """
        Used to connect substructures.
        """
        pass
        # self._graph.add_edge(connection)

    @interface.abstractclassmethod
    def set_variables(self, factors):
        """
        Used by the study and problem to set the first element's factor variables
        """
        pass

    @interface.abstractclassmethod
    def get_variables(self):
        """
        Used by the problem and solver to get the sate of the model
        """
        pass

    # @interface.abstractclassmethod
    # def get_response(self):
    #     """
    #     Used by the study and problem to get the last element's response variables
    #     """
    #     pass

    @interface.abstractclassmethod
    def run(self):
        """
        Used to run the elements of the structure in the appropriate sequence
        """
        pass

# class AbstractElement(interface.ABC):
#     """
#     The nodes of a Structure's graph. These are containiners for either Kernels or Studies.
#     """

#     @interface.abstractclassmethod
#     def run(self):
#         pass

#     @interface.abstractclassmethod
#     def set_variables(self, variables):
#         pass

#     @interface.abstractclassmethod
#     def get_variables(self):
#         pass

#     @interface.abstractclassmethod
#     def get_response(self):
#         pass