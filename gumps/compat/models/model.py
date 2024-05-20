# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from gumps.compat.abstractclasses import AbstractModel
from gumps.compat.common import Container

class Model(Container, AbstractModel):
    """
    A base class for models, with an empty variable registry and an equation 
    hat has no variables and returns nothing.
    """

    def equation(self):
        pass



