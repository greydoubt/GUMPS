# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import abc as interface

class AbstractEngine(interface.ABC):
    """
    The interface for any engine. Any concrete engine must inherit from this.
    The engine is a low-level interface to any software that provides
    mathematical modeling and numerical solution capabilities.
    The engine can instantiate a model and supports at least one kernel "primitive"
    (i.e., usually a simulation-like primitive).
    This class follows the Adaptor and Bridge design patterns.
    This corresponds to the WarmDrive sub-system in Cosmos.
    """

    @interface.abstractclassmethod
    def set_variables(self, kernel, source_variables):
        pass

    @interface.abstractclassmethod
    def get_variables(self, kernel):
        pass

    @interface.abstractclassmethod
    def initialize(self):
        """
        Used to initialize any requirements for the engine before running a study.
        """
        pass

    @interface.abstractclassmethod
    def run(self, kernel):
        """
        Used to call the engine to execute until a complete solution is returned, i.e. this is blocking.
        :kernel: the kernel from which this engine was called. The kernel reference will give access to problem variables
        """
        pass