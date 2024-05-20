# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#This is a simple nd sphere kernel suitable for use testing interfaces,
#optimization, parameter estimation, and uncertain quanfication

from gumps.kernels.kernel import AbstractKernel
import numpy as np
import attrs

@attrs.define
class SphereState:
    "states for the SphereKernel"
    n: int
    x_0: float
    x_1: float
    x_2: float
    x_3: float
    total: float = 0.0


@attrs.define
class SphereStateNanException(SphereState):
    "states for the SphereKernel"
    nan_trigger: float = 0.0
    exception_trigger: float = 0.0
    nan_lower_bound: float = 0.0
    nan_upper_bound: float = 0.0
    exception_lower_bound: float = 0.0
    exception_upper_bound: float = 0.0

class SphereKernel(AbstractKernel):
    """This is a sphere kernel. It is a trivial nd optimization problem """

    def user_defined_function(self, variables: SphereState):
        "compute the sphere function"
        total = 0.0
        for i in range(int(variables.n)):
            total += (getattr(variables, f'x_{i}') - self.cache[f'a_{i}'])**2.0
        variables.total = total

    def initialize(self):
        """setup datastrucures that are needed and store them in self.cache
        For the purpose of a test all of the a variables are being treated
        as expensive data structures and stored in the cache"""
        n = self.model_variables['n']
        for i in range(int(n)):
            self.cache[f'a_{i}'] = self.model_variables[f'a_{i}']

    def get_state_class(self) -> SphereState:
        "get the state class"
        return SphereState

class SphereKernelNanException(SphereKernel):
    """This is a sphere kernel that can return NaNs"""

    def get_state_class(self) -> SphereStateNanException:
        "get the state class"
        return SphereStateNanException

    def user_defined_function(self, variables: SphereStateNanException):
        super().user_defined_function(variables)
        if variables.nan_lower_bound <= variables.nan_trigger <= variables.nan_upper_bound:
            variables.total = np.nan

        if variables.exception_lower_bound <= variables.exception_trigger <= variables.exception_upper_bound:
            raise RuntimeError("Exception raised")
