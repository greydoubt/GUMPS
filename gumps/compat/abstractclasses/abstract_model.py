# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import abc as interface

class AbstractModel(interface.ABC):
    """
    The interface for any model. Any concrete model must inherit from this.
    The model is the standard representation of the system of equations to be solved.
    A model cannot be solved per se; it is the model in the context of a study that can.
    It is important to distinguish specifications that correspond to the model formulation,
    and those that belong to the problem formulation.
    This class follows the Strategy design pattern.
    This corresponds to the Stellar sub-system in Cosmos, although the relationship is not obvious ATM
    """

    @interface.abstractclassmethod
    def set_variables(self, source_variables):
        pass

    @interface.abstractclassmethod
    def get_variables(self):
        pass

    @interface.abstractclassmethod
    def equation(self,t,y,arg1):
        """
        Contains the equations of this model in the form expected by the engines which will execute it.
        """
        pass




