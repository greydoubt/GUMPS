# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implement the Ackley function as a kernel. https://en.wikipedia.org/wiki/Ackley_function"""

import logging

import numpy as np

import gumps.kernels.kernel

import attrs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class AckleyFirstTermState:
    "states for the AckleyFirstTermKernel"
    a: float
    b: float
    x: np.ndarray = attrs.field(converter=np.array)
    term1: float = 0.0

class AckleyFirstTerm(gumps.kernels.kernel.AbstractKernel):
    "compute the first term of the Ackley function"

    def user_defined_function(self, variables: AckleyFirstTermState) -> None:
        "compute the function"
        total = np.sum(variables.x**2)
        term1 = (-variables.a) * np.exp( - variables.b * np.sqrt(total/len(variables.x)))
        variables.term1 = term1

    def get_state_class(self) -> AckleyFirstTermState:
        "get the state class"
        return AckleyFirstTermState


@attrs.define
class AckleySecondTermState:
    "states for the AckleySecondTermKernel"
    c: float
    x: np.ndarray = attrs.field(converter=np.array)
    term2: float = 0.0

class AckleySecondTerm(gumps.kernels.kernel.AbstractKernel):
    "compute the second term of the Ackley function"

    def user_defined_function(self, variables: object) -> None:
        "compute the function"
        mean = np.mean(np.cos(variables.c*variables.x))
        term2 = - np.exp(mean)
        variables.term2 = term2

    def get_state_class(self) -> AckleySecondTermState:
        "get the state class"
        return AckleySecondTermState


@attrs.define
class AckleyFunctionState:
    "states for the AckleyFunctionKernel"
    a: float
    term1: float
    term2: float
    total: float = 0.0

class AckleyFunction(gumps.kernels.kernel.AbstractKernel):
    "compute the Ackley function by depending on the first and second term"

    def user_defined_function(self, variables: AckleyFunctionState) -> None:
        "compute the function"
        variables.total = variables.term1 + variables.term2 + variables.a + np.exp(1)

    def get_state_class(self) -> AckleyFunctionState:
        "get the state class"
        return AckleyFunctionState
