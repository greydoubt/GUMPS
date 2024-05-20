# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import abc as interface

class AbstractProblem(interface.ABC):
    """
    The interface for any problem. Any concrete problem must inherit from this.
    It represents a generic problem, that is specified entirely by some problem data.
    Its single responsibility is to (know how to) apply itself into the engine.
    This class follows the Bridge and Builder design patterns.
    This corresponds to the Singularity sub-system in Cosmos.
    """

    @interface.abstractclassmethod
    def stop_condition(self, state):
        """
        The test applied at each step of the engine to determine whether the study is complete. The study is complete when this function returns True.
        :return: True for complete study, False otherwise
        """
        pass