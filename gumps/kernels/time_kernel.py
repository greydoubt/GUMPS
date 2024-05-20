# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement a simple kernel that returns a time-series"""

import logging
from typing import Type
import numpy as np
import attrs

import gumps.kernels.kernel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class TimeKernelState:
    "states for the TimeKernel"
    time: float
    a: float
    b: float
    c: float
    d: float
    ta: float = 0.0
    tb: float = 0.0
    tc: float = 0.0
    td: float = 0.0

class TimeKernelExample(gumps.kernels.kernel.AbstractKernel):
    "create an example kernel that returns a time-series"

    def user_defined_function(self, variables: TimeKernelState) -> None:
        "compute the function"
        if variables.time < 0:
            raise ValueError("time must be greater than 0")

        if variables.a < 0:
            raise ValueError("a must be greater than 0")

        if variables.b < 0:
            variables.b = np.nan

        variables.ta = variables.a * variables.time
        variables.tb = variables.b * variables.time
        variables.tc = variables.c * variables.time
        variables.td = variables.d * variables.time

    def get_state_class(self) -> Type:
        "return the state class"
        return TimeKernelState


@attrs.define
class SimpleTimeKernelState:
    "states for the SimpleTimeKernel"
    time: float = 0.0
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0


class SimpleJupyterTimeKernel(gumps.kernels.kernel.AbstractKernel):
    "create an example kernel that returns a time-series (needed for jupyter notebook tutorial so the function is pickleable)"

    def user_defined_function(self, variables: SimpleTimeKernelState) -> None:
        "compute the function"
        variables.c1 = variables.a * np.sin(variables.time) + variables.c
        variables.c2 = variables.b * np.cos(variables.time) + variables.c
        variables.c3 = variables.a * np.sin(variables.time) * variables.b * np.cos(variables.time) + variables.c


    def get_state_class(self) -> SimpleTimeKernelState:
        "return the state class"
        return SimpleTimeKernelState
