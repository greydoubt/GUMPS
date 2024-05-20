# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import abc as interface


class AbstractTransform(interface.ABC):
    """
    A data transformation, typically used to transform data between the bases expected by two connected models.
    """

    @interface.abstractclassmethod
    def forward_transform(data):
        return data


        

class AbstractBidirectionalTransform(AbstractTransform):
    """
    A data transformation, typically used to transform data between the bases expected by two connected models.
    """
    
    @interface.abstractclassmethod
    def backward_transform(data):
        return data