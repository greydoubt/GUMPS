# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np

import gumps.kernels.kernel
import gumps.kernels.ackley_kernel

import attrs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@attrs.define
class AckleyCompleteState:
    "states for the AckleyCompleteKernel"
    a: float
    b: float
    c: float
    x: np.ndarray = attrs.field(converter=np.array)
    term1: float = 0.0
    term2: float = 0.0
    total: float = 0.0

class AckleyCompleteKernel(gumps.kernels.kernel.AbstractKernel):
    def user_defined_function(self, variables: AckleyCompleteState) -> None:
        "compute the function"
        sum_squared = np.sum(variables.x**2)
        variables.term1 = (-variables.a) * np.exp( - variables.b * np.sqrt(sum_squared/len(variables.x)))
        mean = np.mean(np.cos(variables.c*variables.x))
        variables.term2 = - np.exp(mean)
        variables.total = variables.term1 + variables.term2 + variables.a + np.exp(1)

    def get_state_class(self) -> AckleyCompleteState:
        "get the state class"
        return AckleyCompleteState

class AckleyCompleteKernelAlternate(gumps.kernels.kernel.AbstractKernel):
    "implement a complete ackley kernel by calling the sub-kernels"

    def __init__(self, model_variables:dict|None = None):
        "Init that takes model variables as input"
        self.ackley_first_term = gumps.kernels.ackley_kernel.AckleyFirstTerm(model_variables)
        self.ackley_second_term = gumps.kernels.ackley_kernel.AckleySecondTerm(model_variables)
        self.ackely_function = gumps.kernels.ackley_kernel.AckleyFunction(model_variables)
        super().__init__(model_variables)

    def user_defined_function(self, variables: AckleyCompleteState) -> None:
        "compute the function"
        self.ackley_first_term.user_defined_function(variables)
        self.ackley_second_term.user_defined_function(variables)
        self.ackely_function.user_defined_function(variables)

    def get_state_class(self) -> AckleyCompleteState:
        "get the state class"
        return AckleyCompleteState

